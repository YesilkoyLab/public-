"""Exploratory LDA feature interpretation; one specimen per ratio.

Requirements: numpy pandas scikit-learn matplotlib
Run from the same working directory as the original script, or edit DATA_DIR.
No inferential p-values or independent-specimen validation are computed.
"""

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler


# ==================== Settings ====================
NAME = "03272026"
DATA_DIR = Path("../res-15/03272026")  # Relative to the working directory.
OUTPUT_DIR = DATA_DIR / "lda_feature_interpretation"
GLCM_FILE = DATA_DIR / f"{NAME}_glcm_patch32_offset5_15slice.csv"
MORPH_FILE = DATA_DIR / f"{NAME}_patch32_features_v2.csv"
N_PER_GROUP = 256
RANDOM_STATE = 42
TOP_N = 10
SHOW_PLOTS = False
DPI = 300

# Add numeric metadata here so it is not accidentally analysed as a feature.
EXCLUDE_COLUMNS = ["file_name", "x", "y", "label"]
MERGE_KEYS = ["file_name", "x", "y"]
DESIRED_ORDER = [
    "1000", "9010SER", "9010PSER", "8020SER", "8020PSER",
    "7030SER", "7030PSER", "6040SER", "6040PSER",
]


def extract_label(file_name):
    return re.sub(r"_\d+$", "", str(file_name).strip()).upper()


def save_figure(fig, stem, output_dir):
    fig.savefig(output_dir / f"{stem}.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def load_and_prepare(glcm_file, morph_file, n_per_group, random_state):
    glcm = pd.read_csv(glcm_file)
    morph = pd.read_csv(morph_file)
    for name, frame in [("GLCM", glcm), ("Morphology", morph)]:
        missing = set(MERGE_KEYS) - set(frame.columns)
        if missing:
            raise ValueError(f"{name}: missing merge keys {sorted(missing)}")
        if frame[MERGE_KEYS].isna().any().any():
            raise ValueError(f"{name}: missing values in merge keys")
        if frame.duplicated(MERGE_KEYS).any():
            raise ValueError(
                f"{name}: repeated (file_name, x, y). Check whether slice/z "
                "must also be a merge key; do not silently multiply rows."
            )

    # Preserve the original rule: morphology wins for duplicate feature names.
    duplicates = [c for c in glcm if c in morph and c not in MERGE_KEYS]
    print("Shared columns retained from morphology:", duplicates)
    merged = glcm.drop(columns=duplicates).merge(
        morph, on=MERGE_KEYS, how="inner", validate="one_to_one"
    )
    print(f"Input rows: GLCM={len(glcm)}, morphology={len(morph)}; matched={len(merged)}")
    if merged.empty:
        raise ValueError("No matched patches. Check file_name and coordinates.")
    merged["label"] = merged["file_name"].map(extract_label)
    labels = merged["label"].unique().tolist()
    order = [g for g in DESIRED_ORDER if g in labels]
    order += sorted(set(labels) - set(order))
    if len(order) < 2:
        raise ValueError("At least two ratio labels are required.")
    print("Available patches per ratio:\n", merged["label"].value_counts())
    groups = []
    for group in order:
        subset = merged.loc[merged["label"] == group]
        if len(subset) < n_per_group:
            raise ValueError(f"{group}: {len(subset)} patches; need {n_per_group}.")
        groups.append(subset.sample(n=n_per_group, random_state=random_state))
    sampled = pd.concat(groups, ignore_index=True)

    columns = [c for c in sampled if c not in EXCLUDE_COLUMNS]
    raw = sampled[columns].apply(pd.to_numeric, errors="coerce")
    raw = raw.replace([np.inf, -np.inf], np.nan)
    all_missing = raw.columns[raw.isna().all()].tolist()
    raw = raw.drop(columns=all_missing)
    if raw.shape[1] == 0:
        raise ValueError("No numeric features remain.")
    missing_counts = raw.isna().sum()
    filled = raw.fillna(raw.median())
    constant = filled.columns[filled.nunique() <= 1].tolist()
    filled = filled.drop(columns=constant)
    raw = raw[filled.columns]
    if filled.shape[1] == 0:
        raise ValueError("All numeric features are constant.")
    report = pd.DataFrame({
        "Feature": columns,
        "missing_or_nonfinite_count": [
            n_per_group * len(order) if c in all_missing else int(missing_counts[c])
            for c in columns
        ],
        "status": [
            "dropped_all_missing_or_nonnumeric" if c in all_missing else
            "dropped_constant" if c in constant else "used"
            for c in columns
        ],
    })
    return sampled, raw, filled, order, report


def fit_and_interpret(features, labels):
    scaler = StandardScaler()
    z = scaler.fit_transform(features)
    n_components = min(2, features.shape[1], len(np.unique(labels)) - 1)
    lda = LinearDiscriminantAnalysis(solver="svd", n_components=n_components)
    scores = lda.fit_transform(z, labels)
    if scores.shape[1] == 0:
        raise ValueError("No discriminant axis was estimable from these features.")
    weights = lda.scalings_[:, :scores.shape[1]].copy()

    # Axis signs are arbitrary. Orient the largest-magnitude weight positively
    # so every exported score, weight, and correlation uses the same convention.
    for axis in range(scores.shape[1]):
        pivot = np.argmax(np.abs(weights[:, axis]))
        sign = -1.0 if weights[pivot, axis] < 0 else 1.0
        weights[:, axis] *= sign
        scores[:, axis] *= sign
    np.testing.assert_allclose(
        (z - lda.xbar_) @ weights, scores, rtol=1e-7, atol=1e-8
    )

    table = pd.DataFrame({"Feature": features.columns})
    for axis in range(scores.shape[1]):
        ld = f"LD{axis + 1}"
        # Total-sample Pearson correlations, NOT causal or unique contributions.
        centered_scores = scores[:, axis] - scores[:, axis].mean()
        centered_z = z - z.mean(axis=0)
        denominator = np.sqrt(
            np.sum(centered_z ** 2, axis=0) * np.sum(centered_scores ** 2)
        )
        correlations = np.divide(
            centered_z.T @ centered_scores, denominator,
            out=np.full(features.shape[1], np.nan), where=denominator > 0,
        )
        table[f"{ld}_weight"] = weights[:, axis]
        table[f"{ld}_abs_weight"] = np.abs(weights[:, axis])
        table[f"{ld}_weight_rank"] = table[f"{ld}_abs_weight"].rank(
            ascending=False, method="min"
        ).astype(int)
        table[f"{ld}_total_correlation"] = np.clip(correlations, -1, 1)
        table[f"{ld}_abs_total_correlation"] = np.abs(correlations)
    preprocessing = pd.DataFrame({
        "Feature": features.columns,
        "imputation_median": features.median().values,
        "standardization_mean": scaler.mean_,
        "standardization_scale": scaler.scale_,
    })
    return lda, scores, table, preprocessing


def plot_ranked_features(table, column, title, xlabel, output_dir, stem):
    selected = table.loc[table[column].abs().nlargest(TOP_N).index]
    selected = selected.sort_values(column)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.42 * len(selected) + 1.5)))
    ax.barh(selected["Feature"], selected[column], color=[
        "#4784AA" if value >= 0 else "#D88776" for value in selected[column]
    ])
    ax.axvline(0, color="0.3", linewidth=0.8)
    ax.set(xlabel=xlabel, title=title)
    if "correlation" in column:
        ax.set_xlim(-1.05, 1.05)
    fig.tight_layout()
    save_figure(fig, stem, output_dir)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    sampled, raw, features, order, cleaning = load_and_prepare(
        GLCM_FILE, MORPH_FILE, N_PER_GROUP, RANDOM_STATE
    )
    labels = sampled["label"].to_numpy()
    lda, scores, table, preprocessing = fit_and_interpret(features, labels)
    n_axes = scores.shape[1]
    table.to_csv(OUTPUT_DIR / f"{NAME}_all_feature_axis_weights_correlations.csv", index=False)
    cleaning.to_csv(OUTPUT_DIR / f"{NAME}_feature_cleaning.csv", index=False)
    preprocessing.to_csv(OUTPUT_DIR / f"{NAME}_preprocessing.csv", index=False)
    projection = sampled[MERGE_KEYS + ["label"]].copy()
    for axis in range(n_axes):
        projection[f"LD{axis + 1}"] = scores[:, axis]
    projection.to_csv(OUTPUT_DIR / f"{NAME}_LDA_projection.csv", index=False)
    features.corr().to_csv(OUTPUT_DIR / f"{NAME}_feature_correlation_matrix.csv")
    pd.DataFrame({"Axis": [f"LD{i + 1}" for i in range(n_axes)],
                  "discriminant_ratio": lda.explained_variance_ratio_[:n_axes]}).to_csv(
        OUTPUT_DIR / f"{NAME}_axis_discriminant_ratios.csv", index=False
    )
    tab20 = plt.get_cmap("tab20").colors
    palette = dict(zip(DESIRED_ORDER, [tab20[i] for i in [14, 0, 1, 2, 3, 4, 5, 6, 7]]))
    for i, group in enumerate(order):
        palette.setdefault(group, tab20[i % len(tab20)])

    fig, ax = plt.subplots(figsize=(10, 7))
    for group in order:
        mask = labels == group
        ax.scatter(scores[mask, 0], scores[mask, 1] if n_axes > 1 else
                   np.zeros(mask.sum()), s=18, alpha=0.65,
                   color=palette[group], label=group, edgecolors="none")
    ax.set_xlabel(f"LD1 ({lda.explained_variance_ratio_[0]:.1%} of discriminant ratio)")
    ax.set_ylabel(f"LD2 ({lda.explained_variance_ratio_[1]:.1%} of discriminant ratio)"
                  if n_axes > 1 else "No second discriminant axis")
    ax.set_title("SHG LDA: exploratory separation of measured specimens")
    ax.legend(title="Ratio", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    save_figure(fig, f"{NAME}_LDA_scatter", OUTPUT_DIR)

    top_features = []
    for axis in range(n_axes):
        ld = f"LD{axis + 1}"
        ranked = table.sort_values(f"{ld}_abs_weight", ascending=False)
        ranked.head(TOP_N).to_csv(OUTPUT_DIR / f"{NAME}_{ld}_top_weights.csv", index=False)
        corr_ranked = table.sort_values(f"{ld}_abs_total_correlation", ascending=False)
        corr_ranked.head(TOP_N).to_csv(
            OUTPUT_DIR / f"{NAME}_{ld}_top_total_correlations.csv", index=False
        )
        plot_ranked_features(table, f"{ld}_weight", f"{ld}: top standardized-input weights",
                             f"Weight in {ld} projection", OUTPUT_DIR, f"{NAME}_{ld}_top_weights")
        plot_ranked_features(table, f"{ld}_total_correlation", f"{ld}: top feature-score correlations",
                             f"Total-sample Pearson r with {ld}", OUTPUT_DIR,
                             f"{NAME}_{ld}_top_total_correlations")
        top_features.extend(ranked.head(TOP_N)["Feature"])
        top_features.extend(corr_ranked.head(TOP_N)["Feature"])
        print(f"\n{ld}: top weights\n", ranked.head(TOP_N).to_string(index=False))

    # Distributions and summaries use observed values, not median-imputed values.
    top_features = list(dict.fromkeys(top_features))
    distributions = OUTPUT_DIR / "top_feature_distributions"
    distributions.mkdir(exist_ok=True)
    summaries = []
    for index, feature in enumerate(top_features, start=1):
        arrays = [raw.loc[labels == group, feature].dropna().to_numpy() for group in order]
        valid = [(i + 1, values) for i, values in enumerate(arrays) if len(values)]
        fig, ax = plt.subplots(figsize=(11, 5))
        boxes = ax.boxplot([v for _, v in valid], positions=[i for i, _ in valid],
                           widths=0.6, patch_artist=True, showfliers=True)
        for box, (position, _) in zip(boxes["boxes"], valid):
            box.set_facecolor(palette[order[position - 1]])
            box.set_alpha(0.75)
        ax.set_xticks(range(1, len(order) + 1))
        ax.set_xticklabels(order, rotation=40, ha="right")
        ax.set_ylabel(feature)
        ax.set_title(f"{feature}: observed patch distributions")
        fig.text(0.5, 0.01, "One specimen per ratio; boxes describe patches, not independent specimens.",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", feature)[:100]
        save_figure(fig, f"{index:02d}_{safe}", distributions)
        for group, values in zip(order, arrays):
            summaries.append({"Feature": feature, "Group": group,
                              "n_observed_patches": len(values),
                              "mean": np.mean(values) if len(values) else np.nan,
                              "SD_across_patches": np.std(values, ddof=1) if len(values) > 1 else np.nan,
                              "median": np.median(values) if len(values) else np.nan})
    pd.DataFrame(summaries).to_csv(OUTPUT_DIR / f"{NAME}_top_feature_group_summaries.csv", index=False)
    (OUTPUT_DIR / "INTERPRETATION.txt").write_text(
        "One independently prepared specimen per ratio; 256 sampled patches per ratio.\n"
        "Results are exploratory descriptions of measured specimens. No p-values are computed.\n"
        "Weights use globally standardized, median-imputed inputs and lda.scalings_.\n"
        "Each LD axis is oriented to make its largest absolute weight positive.\n"
        "Absolute weights rank projection terms, not unique or causal contributions.\n"
        "Total correlations are Pearson correlations of each feature with fitted LD scores.\n"
        "Correlations and weights use the same training data; they are not independent validation.\n"
        "Correlated/redundant features can have unstable weights. Inspect the correlation matrix.\n"
        "No ranking stability across independent specimens can be established with this design.\n"
        "Discriminant ratios are not percentages of total original-feature variance.\n"
        "Boxplots: median, Q1-Q3, whiskers to 1.5 IQR, and outlying patches.\n"
        "Boxplots and group summaries omit missing values instead of showing imputed values.\n",
        encoding="utf-8",
    )
    print(f"\nSaved analysis to: {OUTPUT_DIR.resolve()}")
    warnings.warn("One specimen per ratio: interpret separation as exploratory, not a replicated ratio effect.")


if __name__ == "__main__":
    main()
