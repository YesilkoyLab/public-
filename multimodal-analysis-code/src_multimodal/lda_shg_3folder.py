import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_selection import f_classif
from matplotlib.lines import Line2D


# =========================================================
# Settings
# =========================================================
# folder_configs = [
#     {
#         "name": "replicate1",
#         "base_path": "../res/03252026",
#         "glcm_csv": "03252026_glcm_patch64_offset5.csv",
#         "morph_csv": "03252026_patch64_features.csv",
#         "fw_only": False
#     },
#     {
#         "name": "replicate2",
#         "base_path": "../res/03272026",
#         "glcm_csv": "03272026_glcm_patch32_offset5.csv",
#         "morph_csv": "03272026_patch32_features.csv",
#         "fw_only": False
#     },
#     {
#         "name": "replicate3",
#         "base_path": "../res/09302025",
#         "glcm_csv": "09302025_glcm_patch32_offset5.csv",
#         "morph_csv": "09302025_patch32_features.csv",
#         "fw_only": True
#     }
# ]

folder_configs = [
    {
        "name": "replicate1",
        "base_path": "/Users/tianyizheng/Library/Mobile Documents/com~apple~CloudDocs/Documents/pyws/spec/res_review/SHG_data/BIO-REP-1",
        "glcm_csv": "BIO-REP-1_glcm_patch32_offset5_15slice.csv",
        "morph_csv": "BIO-REP-1_patch32_features.csv",
        "fw_only": False
    },
    {
        "name": "replicate2",
        "base_path": "/Users/tianyizheng/Library/Mobile Documents/com~apple~CloudDocs/Documents/pyws/spec/res_review/SHG_data/BIO-REP-2",
        "glcm_csv": "BIO-REP-2_glcm_patch32_offset5_15slice.csv",
        "morph_csv": "BIO-REP-2_patch32_features.csv",
        "fw_only": False
    },
    {
        "name": "replicate3",
        "base_path": "/Users/tianyizheng/Library/Mobile Documents/com~apple~CloudDocs/Documents/pyws/spec/res_review/SHG_data/BIO-REP-3",
        "glcm_csv": "BIO-REP-3_glcm_patch32_offset5_15slice.csv",
        "morph_csv": "BIO-REP-3_patch32_features.csv",
        "fw_only": False
    }
]

name = "three_replicates_PCA_LDA_LOGO"
save_path = "../res/three_replicates_pca_lda_logo_test_newdata"
os.makedirs(save_path, exist_ok=True)

N_PER_GROUP = 128
RANDOM_STATE = 42

# PCA setting:
# use float (e.g. 0.95) to keep enough PCs explaining 95% variance
# or set fixed integer like 20
PCA_N_COMPONENTS = 0.95


# =========================================================
# Color / marker maps
# =========================================================
def build_ratio_color_map():
    tab20 = plt.get_cmap("tab20").colors
    return {
        "1000": tab20[14],
        "9010SER": tab20[0],
        "9010PSER": tab20[1],
        "8020SER": tab20[2],
        "8020PSER": tab20[3],
        "7030SER": tab20[4],
        "7030PSER": tab20[5],
        "6040SER": tab20[6],
        "6040PSER": tab20[7],
    }

ratio_color_map = build_ratio_color_map()

folder_marker_map = {
    "replicate1": "o",
    "replicate2": "^",
    "replicate3": "s"
}


# =========================================================
# Helpers
# =========================================================
def extract_label(file_name, fw_only=False):
    """Normalize labels; require -fw only for the 09302025 batch.

    Other batches: 9010ser_2 -> 9010SER.
    fw_only=True: 9010ser-fw_2 or 9010ser_2-fw -> 9010SER;
    backward and directionless observations are excluded in that batch.
    """
    if not fw_only:
        return re.sub(r"_\d+$", "", str(file_name)).upper()
    text = str(file_name).strip().upper()
    text = re.sub(r"\.(?:TIFF?|CSV|NPY)$", "", text)
    if re.search(r"-BW(?=$|[_-])", text):
        return None
    match = re.fullmatch(r"(.+?)-FW(?:_\d+)?", text)
    if match is None:
        return None
    return re.sub(r"_\d+$", "", match.group(1))


def keep_forward_rows(df, source_name):
    if "file_name" not in df.columns:
        raise ValueError(f"{source_name}: missing file_name column")
    keep = df["file_name"].apply(lambda value: extract_label(value, fw_only=True)).notna()
    print(f"{source_name}: retained {int(keep.sum())}/{len(df)} forward rows")
    result = df.loc[keep].copy()
    if result.empty:
        raise ValueError(f"{source_name}: no -fw observations found; check file_name values")
    return result


def p_to_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


def load_and_merge_one_folder(cfg):
    glcm_path = os.path.join(cfg["base_path"], cfg["glcm_csv"])
    morph_path = os.path.join(cfg["base_path"], cfg["morph_csv"])

    glcm_df = pd.read_csv(glcm_path)
    morph_df = pd.read_csv(morph_path)

    # Only 09302025 has direction suffixes. Preserve original merge keys.
    fw_only = cfg.get("fw_only", False)
    if fw_only:
        glcm_df = keep_forward_rows(glcm_df, glcm_path)
        morph_df = keep_forward_rows(morph_df, morph_path)

    print(f"\n=== {cfg['name']} ===")
    print("glcm_df shape:", glcm_df.shape)
    print("morph_df shape:", morph_df.shape)

    merge_keys = ["file_name", "x", "y"]

    duplicate_cols = [c for c in glcm_df.columns if c in morph_df.columns and c not in merge_keys]
    print("Duplicate columns:", duplicate_cols)

    # keep morphology version if duplicate exists
    glcm_df_clean = glcm_df.drop(columns=duplicate_cols, errors="ignore")

    merged_df = pd.merge(
        glcm_df_clean,
        morph_df,
        on=merge_keys,
        how="inner"
    )

    if merged_df.empty:
        raise ValueError(f"{cfg['name']}: no matching rows between feature tables")
    merged_df["label"] = merged_df["file_name"].apply(
        lambda value: extract_label(value, fw_only=fw_only)
    )
    merged_df["folder"] = cfg["name"]

    print("merged_df shape:", merged_df.shape)
    print("Label counts:")
    print(merged_df["label"].value_counts())

    return merged_df


# =========================================================
# Load all replicates
# =========================================================
all_merged = []
for cfg in folder_configs:
    all_merged.append(load_and_merge_one_folder(cfg))

merged_df = pd.concat(all_merged, axis=0, ignore_index=True)
print("\nCombined merged_df shape:", merged_df.shape)


# =========================================================
# Label order
# =========================================================
desired_order = [
    "1000",
    "9010SER",
    "9010PSER",
    "8020SER",
    "8020PSER",
    "7030SER",
    "7030PSER",
    "6040SER",
    "6040PSER"
]

existing_labels = merged_df["label"].unique().tolist()
label_order = [g for g in desired_order if g in existing_labels] + \
              [g for g in sorted(existing_labels) if g not in desired_order]

print("Final label order:", label_order)


# =========================================================
# Randomly sample same number per group per replicate
# =========================================================
sampled_groups = []

for folder_name in [cfg["name"] for cfg in folder_configs]:
    for group in label_order:
        group_df = merged_df[
            (merged_df["folder"] == folder_name) &
            (merged_df["label"] == group)
        ].copy()

        if len(group_df) == 0:
            continue

        if len(group_df) < N_PER_GROUP:
            raise ValueError(
                f"Folder '{folder_name}', Group '{group}' has only {len(group_df)} rows, fewer than {N_PER_GROUP}."
            )

        sampled_group = group_df.sample(n=N_PER_GROUP, random_state=RANDOM_STATE)
        sampled_groups.append(sampled_group)

final_df = pd.concat(sampled_groups, axis=0).reset_index(drop=True)

print("\nAfter random sampling:")
print(final_df.groupby(["folder", "label"]).size())
print("final_df shape:", final_df.shape)


# =========================================================
# Build feature matrix
# =========================================================
drop_cols = ["file_name", "x", "y", "label", "folder"]
feature_cols = [c for c in final_df.columns if c not in drop_cols]

print("Number of features:", len(feature_cols))

X_full = final_df[feature_cols].apply(pd.to_numeric, errors="coerce")
X_full = X_full.replace([np.inf, -np.inf], np.nan)

if X_full.isna().sum().sum() > 0:
    print("Found NaNs, filling with column median.")
    X_full = X_full.fillna(X_full.median(numeric_only=True))

y_full = final_df["label"].values
rep_full = final_df["folder"].values

unique_reps = [cfg["name"] for cfg in folder_configs]


# =========================================================
# Leave-one-biological-replicate-out CV
# =========================================================
all_test_true = []
all_test_pred = []
fold_results = []
coef_back_list = []
proj_rows = []

for test_rep in unique_reps:
    print(f"\n--- Fold test replicate: {test_rep} ---")

    train_mask = rep_full != test_rep
    test_mask = rep_full == test_rep

    X_train = X_full.loc[train_mask, feature_cols].values
    X_test = X_full.loc[test_mask, feature_cols].values
    y_train = y_full[train_mask]
    y_test = y_full[test_mask]

    # ---------------------------------
    # Standardize using TRAIN ONLY
    # ---------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---------------------------------
    # PCA using TRAIN ONLY
    # ---------------------------------
    pca = PCA(n_components=PCA_N_COMPONENTS, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    print("Train PCA shape:", X_train_pca.shape)
    print("Test PCA shape:", X_test_pca.shape)

    # ---------------------------------
    # LDA
    # ---------------------------------
    n_classes_train = len(np.unique(y_train))
    lda_n_components = min(2, X_train_pca.shape[1], n_classes_train - 1)

    if lda_n_components < 1:
        raise ValueError("Not enough classes for LDA in this fold.")

    lda = LinearDiscriminantAnalysis(n_components=lda_n_components)
    X_train_lda = lda.fit_transform(X_train_pca, y_train)
    X_test_lda = lda.transform(X_test_pca)

    y_pred = lda.predict(X_test_pca)
    acc = accuracy_score(y_test, y_pred)

    print(f"Fold accuracy = {acc:.4f}")

    all_test_true.extend(y_test)
    all_test_pred.extend(y_pred)

    fold_results.append({
        "test_replicate": test_rep,
        "accuracy": acc,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "pca_components": X_train_pca.shape[1],
        "lda_components": lda_n_components
    })

    # ---------------------------------
    # approximate feature importance
    # back-project LDA coefficients to original feature space
    # ---------------------------------
    coef_pca_space = lda.coef_
    coef_original_space = coef_pca_space @ pca.components_   # shape: [n_classes, n_features]

    # average absolute value across classes
    coef_back = np.mean(np.abs(coef_original_space), axis=0)
    coef_back_list.append(coef_back)

    # ---------------------------------
    # save test projection
    # ---------------------------------
    fold_proj = pd.DataFrame({
        "LDA1": X_test_lda[:, 0],
        "label": y_test,
        "folder": test_rep,
        "fold": test_rep
    })

    if lda_n_components > 1:
        fold_proj["LDA2"] = X_test_lda[:, 1]
    else:
        fold_proj["LDA2"] = 0.0

    proj_rows.append(fold_proj)


# =========================================================
# Fold results summary
# =========================================================
fold_results_df = pd.DataFrame(fold_results)
fold_results_df.to_csv(os.path.join(save_path, f"{name}_LOGO_fold_results.csv"), index=False)

overall_acc = accuracy_score(all_test_true, all_test_pred)
acc_mean = fold_results_df["accuracy"].mean()
acc_std = fold_results_df["accuracy"].std(ddof=1) if len(fold_results_df) > 1 else 0.0

print("\nOverall LOGO accuracy:", overall_acc)
print("Mean fold accuracy ± std:", acc_mean, acc_std)

summary_df = pd.DataFrame({
    "overall_accuracy": [overall_acc],
    "mean_fold_accuracy": [acc_mean],
    "std_fold_accuracy": [acc_std],
    "n_replicates": [len(unique_reps)],
    "n_samples": [len(final_df)],
    "n_features": [len(feature_cols)]
})
summary_df.to_csv(os.path.join(save_path, f"{name}_summary.csv"), index=False)


# =========================================================
# Confusion matrix
# =========================================================
cm = confusion_matrix(all_test_true, all_test_pred, labels=label_order)
cm_df = pd.DataFrame(cm, index=label_order, columns=label_order)
cm_df.to_csv(os.path.join(save_path, f"{name}_LOGO_confusion_matrix.csv"))

fig, ax = plt.subplots(figsize=(8, 7))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_order)
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
plt.title(f"LOGO Confusion Matrix\nOverall accuracy = {overall_acc:.3f}")
plt.tight_layout()
plt.savefig(os.path.join(save_path, f"{name}_LOGO_confusion_matrix.png"), dpi=300, bbox_inches="tight")
plt.show()


# =========================================================
# Combined PCA→LDA scatter using test projections from all folds
# =========================================================
lda_df = pd.concat(proj_rows, axis=0, ignore_index=True)
lda_df.to_csv(os.path.join(save_path, f"{name}_PCA_LDA_LOGO_projection.csv"), index=False)

plt.figure(figsize=(12, 8))

for rep in unique_reps:
    for group in label_order:
        mask = (lda_df["folder"] == rep) & (lda_df["label"] == group)
        if np.sum(mask) == 0:
            continue

        plt.scatter(
            lda_df.loc[mask, "LDA1"],
            lda_df.loc[mask, "LDA2"],
            color=ratio_color_map.get(group, "gray"),
            marker=folder_marker_map.get(rep, "o"),
            s=50,
            alpha=0.75,
            edgecolors="none"
        )

plt.xlabel("LDA Component 1", fontsize=16)
plt.ylabel("LDA Component 2", fontsize=16)
plt.title("PCA → LDA with Leave-One-Biological-Replicate-Out CV", fontsize=18)

color_handles = [
    mpatches.Patch(color=ratio_color_map[g], label=g)
    for g in label_order if g in ratio_color_map
]

marker_handles = [
    Line2D([0], [0],
           marker=folder_marker_map.get(rep, "o"),
           color="black",
           linestyle="None",
           markersize=8,
           label=rep)
    for rep in unique_reps
]

legend1 = plt.legend(
    handles=color_handles,
    title="Group",
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    frameon=False
)
plt.gca().add_artist(legend1)

plt.legend(
    handles=marker_handles,
    title="Replicate",
    bbox_to_anchor=(1.05, 0.45),
    loc="upper left",
    frameon=False
)

plt.tight_layout()
plt.savefig(os.path.join(save_path, f"{name}_PCA_LDA_LOGO_scatter.png"), dpi=300, bbox_inches="tight")
plt.show()


# =========================================================
# Feature importance
# =========================================================
mean_importance = np.mean(np.vstack(coef_back_list), axis=0)

importance_df = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": mean_importance
}).sort_values(by="Importance", ascending=False)

importance_df.to_csv(
    os.path.join(save_path, f"{name}_PCA_LDA_feature_importance.csv"),
    index=False
)

print("Feature importance saved.")


# =========================================================
# Optional ANOVA on full dataset
# Note: this is descriptive, not fold-specific
# =========================================================
# For descriptive ANOVA, standardize on full dataset once
scaler_full = StandardScaler()
X_full_scaled = scaler_full.fit_transform(X_full.values)

F_values, p_values = f_classif(X_full_scaled, y_full)

feat_df = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": mean_importance,
    "F_value": F_values,
    "p_value": p_values
}).sort_values(by="Importance", ascending=False)

feat_df.to_csv(
    os.path.join(save_path, f"{name}_PCA_LDA_all_features_importance_pvalues.csv"),
    index=False
)


# =========================================================
# Top 15 features plot
# =========================================================
top_df = feat_df.head(15).copy()

# use unsigned importance for ranking and plotting
top_df["Importance_signed"] = top_df["Importance"]
top_df["significance"] = top_df["p_value"].apply(p_to_stars)
top_df = top_df.sort_values("Importance_signed")

colors = ["skyblue"] * len(top_df)

plt.figure(figsize=(12, 5))
bars = plt.barh(top_df["Feature"], top_df["Importance_signed"], color=colors)
plt.xlabel("Mean abs back-projected LDA coefficient")
plt.title("Top 15 PCA→LDA Feature Importance", fontsize=18)

for bar, p_val, star in zip(bars, top_df["p_value"], top_df["significance"]):
    width = bar.get_width()
    text = star if star else f"{p_val:.2e}"
    offset = 0.02 * max(top_df["Importance_signed"].max(), 1)

    plt.text(
        width + offset,
        bar.get_y() + bar.get_height() / 2,
        text,
        va="center",
        ha="left",
        fontsize=9
    )

legend_elements = [
    mpatches.Patch(facecolor="none", edgecolor="none", label="* p<0.05"),
    mpatches.Patch(facecolor="none", edgecolor="none", label="** p<0.01"),
    mpatches.Patch(facecolor="none", edgecolor="none", label="*** p<0.001")
]
plt.legend(handles=legend_elements, loc="lower right", frameon=False)

plt.tight_layout()
plt.savefig(os.path.join(save_path, f"{name}_PCA_LDA_Top15_pvalues.png"), dpi=300, bbox_inches="tight")
plt.show()

top_df.to_csv(
    os.path.join(save_path, f"{name}_PCA_LDA_Top15_features_with_pvalues.csv"),
    index=False
)


# =========================================================
# Save outputs
# =========================================================
print("\nSaved:")
print(f"  -> {name}_summary.csv")
print(f"  -> {name}_LOGO_fold_results.csv")
print(f"  -> {name}_LOGO_confusion_matrix.csv")
print(f"  -> {name}_LOGO_confusion_matrix.png")
print(f"  -> {name}_PCA_LDA_LOGO_projection.csv")
print(f"  -> {name}_PCA_LDA_LOGO_scatter.png")
print(f"  -> {name}_PCA_LDA_feature_importance.csv")
print(f"  -> {name}_PCA_LDA_all_features_importance_pvalues.csv")
print(f"  -> {name}_PCA_LDA_Top15_features_with_pvalues.csv")
print(f"  -> {name}_PCA_LDA_Top15_pvalues.png")