import os
import re
import random
import numpy as np
import pandas as pd
import seaborn as sns
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_selection import f_classif


# =========================================================
# Settings
# =========================================================
name = "03272026_fused_singleRep_PCA_LDA"

# ---------- SHG ----------
shg_path = "../res/03272026/"
shg_glcm_path = os.path.join(shg_path, "03272026_glcm_patch32_offset5.csv")
shg_morph_path = os.path.join(shg_path, "03272026_patch32_features.csv")

# ---------- SPEC ----------
spectral_root = r"../../spec_analysis/res/Caf2_03252026/2nd"

# ---------- save ----------
save_path = os.path.join(shg_path, "fused_singleRep_PCA_LDA")
os.makedirs(save_path, exist_ok=True)

# ---------- classes ----------
ratios = [
    "1000",
    "9010SER",
    "9010PSER",
    "8020SER",
    "8020PSER",
    "7030SER",
    "7030PSER",
    "6040SER",
    "6040PSER",
]
ratios_upper = [r.upper() for r in ratios]

# one biological replicate, three technical folders
subfolders = ["LMT_1", "LMT_2", "LMT_3"]

# total samples per class after combining all LMT folders
sample_per_group = 128

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# PCA setting
PCA_N_COMPONENTS = 0.95


# =========================================================
# Color map
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


# =========================================================
# Helpers
# =========================================================
def extract_label(file_name):
    """
    1000_1 -> 1000
    9010ser_2 -> 9010SER
    """
    label = re.sub(r'_\d+$', '', str(file_name))
    return label.upper()


def find_case_insensitive_subdir(parent_dir, target_name):
    if not os.path.isdir(parent_dir):
        raise ValueError(f"Directory not found: {parent_dir}")

    target_upper = target_name.upper()
    for item in os.listdir(parent_dir):
        full_path = os.path.join(parent_dir, item)
        if os.path.isdir(full_path) and item.upper() == target_upper:
            return item

    raise ValueError(f"Cannot find folder matching '{target_name}' under: {parent_dir}")


def p_to_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


# =========================================================
# Load SHG
# =========================================================
glcm_df = pd.read_csv(shg_glcm_path)
morph_df = pd.read_csv(shg_morph_path)

print("glcm_df shape:", glcm_df.shape)
print("morph_df shape:", morph_df.shape)

merge_keys = ["file_name", "x", "y"]
duplicate_cols = [c for c in glcm_df.columns if c in morph_df.columns and c not in merge_keys]
print("Duplicate SHG columns:", duplicate_cols)

glcm_df_clean = glcm_df.drop(columns=duplicate_cols, errors="ignore")
merged_shg_df = pd.merge(glcm_df_clean, morph_df, on=merge_keys, how="inner")
merged_shg_df["label"] = merged_shg_df["file_name"].apply(extract_label)

merged_shg_df = merged_shg_df[merged_shg_df["label"].isin(ratios_upper)].copy()
merged_shg_df = merged_shg_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

print("merged_shg_df shape:", merged_shg_df.shape)
print("SHG label counts:")
print(merged_shg_df["label"].value_counts())


# =========================================================
# Sample SHG per class
# =========================================================
shg_selected = []

for ratio in ratios_upper:
    group_df = merged_shg_df[merged_shg_df["label"] == ratio].copy()

    if len(group_df) < sample_per_group:
        raise ValueError(
            f"Not enough SHG patches for {ratio}. Found {len(group_df)}, need {sample_per_group}."
        )

    sampled = group_df.sample(n=sample_per_group, random_state=RANDOM_STATE).reset_index(drop=True)
    shg_selected.append(sampled)

shg_selected_df = pd.concat(shg_selected, axis=0, ignore_index=True)

# enforce class order
ordered_blocks = []
for ratio in ratios_upper:
    ordered_blocks.append(
        shg_selected_df[shg_selected_df["label"] == ratio].reset_index(drop=True)
    )
shg_selected_df = pd.concat(ordered_blocks, axis=0, ignore_index=True)

print("shg_selected_df shape:", shg_selected_df.shape)
print("Selected SHG label counts:")
print(shg_selected_df["label"].value_counts())


# =========================================================
# Load SPEC from all 3 LMT folders, merged into same class
# =========================================================
spectral_rows = []
expected_length = 426

per_folder = sample_per_group // len(subfolders)
remainder = sample_per_group % len(subfolders)

for ratio in ratios:
    matched_ratio_folder = find_case_insensitive_subdir(spectral_root, ratio)

    selected_files = []
    for i, subfolder in enumerate(subfolders):
        folder_path = os.path.join(spectral_root, matched_ratio_folder, subfolder)
        npy_files = sorted(glob(os.path.join(folder_path, "*.npy")))

        need_n = per_folder + (1 if i < remainder else 0)

        if len(npy_files) < need_n:
            raise ValueError(
                f"Not enough spectral files for {ratio} in {folder_path}. "
                f"Found {len(npy_files)}, need {need_n}."
            )

        sampled_files = random.sample(npy_files, need_n)
        selected_files.extend([(f, subfolder) for f in sampled_files])

    for f, subfolder in selected_files:
        spectrum = np.load(f)
        spectrum = np.asarray(spectrum).flatten()

        if len(spectrum) != expected_length:
            raise ValueError(
                f"Skipping {f}: expected length {expected_length}, got {len(spectrum)}"
            )

        if np.any(~np.isfinite(spectrum)):
            raise ValueError(f"Invalid spectral values in {f}")

        spectral_rows.append({
            "file": f,
            "label": ratio.upper(),
            "subfolder": subfolder,
            "spectrum": spectrum
        })

spectral_df = pd.DataFrame(spectral_rows)

print("spectral_df shape:", spectral_df.shape)
print("Spectral label counts:")
print(spectral_df["label"].value_counts())


# =========================================================
# Convert spectral data to feature table
# =========================================================
spec_feature_mat = np.vstack(spectral_df["spectrum"].values)
spec_feature_df = pd.DataFrame(spec_feature_mat)

# rename spectral columns to wavenumbers: 950..1800 step 2
spec_feature_df.columns = [str(w) for w in range(950, 1802, 2)]
spec_feature_df["label"] = spectral_df["label"].values

# enforce same class order
ordered_spec_blocks = []
for ratio in ratios_upper:
    ordered_spec_blocks.append(
        spec_feature_df[spec_feature_df["label"] == ratio].reset_index(drop=True)
    )
spec_feature_df = pd.concat(ordered_spec_blocks, axis=0, ignore_index=True)

print("spec_feature_df shape:", spec_feature_df.shape)


# =========================================================
# Prepare SHG feature table
# =========================================================
shg_feature_df = shg_selected_df.drop(columns=["file_name", "x", "y"], errors="ignore").copy()

# ensure labels align
if not np.array_equal(spec_feature_df["label"].values, shg_feature_df["label"].values):
    raise ValueError("Spectral labels and SHG labels do not align after sampling/order enforcement.")

# remove duplicated label column before concat
shg_only = shg_feature_df.drop(columns=["label"]).reset_index(drop=True)
spec_only = spec_feature_df.drop(columns=["label"]).reset_index(drop=True)

combined_df = pd.concat([spec_only, shg_only], axis=1)
combined_df["label"] = spec_feature_df["label"].values

print("combined_df shape:", combined_df.shape)
print("Combined label counts:")
print(combined_df["label"].value_counts())


# =========================================================
# Feature matrix
# =========================================================
X = combined_df.drop(columns=["label"]).copy()
y = combined_df["label"].values

X.columns = X.columns.astype(str)
X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)

if X.isna().sum().sum() > 0:
    print("Found NaNs in X. Filling with column median.")
    X = X.fillna(X.median(numeric_only=True))

feature_names = X.columns.tolist()

# optional modality annotation
spec_cols = set(str(w) for w in range(950, 1802, 2))
feature_modality = ["spec" if c in spec_cols else "shg" for c in feature_names]


# =========================================================
# Standardize
# =========================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("X_scaled shape:", X_scaled.shape)


# =========================================================
# PCA
# =========================================================
pca = PCA(n_components=PCA_N_COMPONENTS, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)

print("X_pca shape:", X_pca.shape)
print("Explained variance ratio sum:", np.sum(pca.explained_variance_ratio_))


# =========================================================
# LDA
# =========================================================
n_classes = len(np.unique(y))
lda_n_components = min(2, X_pca.shape[1], n_classes - 1)

if lda_n_components < 1:
    raise ValueError("Need at least 2 classes for LDA.")

lda = LinearDiscriminantAnalysis(n_components=lda_n_components)
X_lda = lda.fit_transform(X_pca, y)

print("X_lda shape:", X_lda.shape)


# =========================================================
# Projection dataframe
# =========================================================
lda_df = pd.DataFrame({
    "LDA1": X_lda[:, 0],
    "Group": y
})

if lda_n_components > 1:
    lda_df["LDA2"] = X_lda[:, 1]
else:
    lda_df["LDA2"] = 0.0

lda_df.to_csv(os.path.join(save_path, f"{name}_PCA_LDA_projection.csv"), index=False)


# =========================================================
# Scatter plot
# =========================================================
palette = {g: ratio_color_map[g] for g in ratios_upper}

plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=lda_df,
    x="LDA1",
    y="LDA2",
    hue="Group",
    hue_order=ratios_upper,
    palette=palette,
    s=60,
    alpha=0.7,
    edgecolor=None
)
plt.xlabel("LDA Component 1")
plt.ylabel("LDA Component 2")
plt.title("PCA → LDA: SHG + Spectral Features")
plt.legend(title="Group", bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(save_path, f"{name}_PCA_LDA_scatter.png"), dpi=300, bbox_inches="tight")
plt.show()


# =========================================================
# Confusion matrix (resubstitution; descriptive only)
# =========================================================
y_pred = lda.predict(X_pca)
cm = confusion_matrix(y, y_pred, labels=ratios_upper)
cm_df = pd.DataFrame(cm, index=ratios_upper, columns=ratios_upper)
cm_df.to_csv(os.path.join(save_path, f"{name}_confusion_matrix.csv"))

fig, ax = plt.subplots(figsize=(8, 7))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ratios_upper)
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(save_path, f"{name}_confusion_matrix.png"), dpi=300, bbox_inches="tight")
plt.show()


# =========================================================
# Feature importance
# back-project LDA coefficients to original feature space
# =========================================================
coef_pca_space = lda.coef_
coef_original_space = coef_pca_space @ pca.components_
lda_importances = np.mean(np.abs(coef_original_space), axis=0)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Modality": feature_modality,
    "Importance": lda_importances
}).sort_values(by="Importance", ascending=False)

importance_df.to_csv(
    os.path.join(save_path, f"{name}_feature_importance.csv"),
    index=False
)

# coefficient heatmap in PCA space
coef_df = pd.DataFrame(lda.coef_)
coef_df.columns = [f"PC{i+1}" for i in range(coef_df.shape[1])]
coef_df.index = [str(c) for c in lda.classes_]

plt.figure(figsize=(14, 4))
sns.heatmap(
    coef_df,
    cmap="vlag",
    center=0,
    annot=False,
    cbar_kws={"label": "LDA Coefficient"}
)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.title("LDA Coefficient Heatmap (PCA space)")
plt.tight_layout()
plt.savefig(os.path.join(save_path, f"{name}_lda_coefficient_heatmap.png"), dpi=300, bbox_inches="tight")
plt.show()


# =========================================================
# ANOVA
# =========================================================
F_values, p_values = f_classif(X_scaled, y)

feat_df = pd.DataFrame({
    "Feature": feature_names,
    "Modality": feature_modality,
    "Importance": lda_importances,
    "F_value": F_values,
    "p_value": p_values
}).sort_values(by="Importance", ascending=False)

feat_df.to_csv(
    os.path.join(save_path, f"{name}_all_features_importance_pvalues.csv"),
    index=False
)


# =========================================================
# Top 20 features
# =========================================================
top_df = feat_df.head(20).copy()
top_df["significance"] = top_df["p_value"].apply(p_to_stars)

# keep unsigned importance for fused display
top_df = top_df.sort_values("Importance")

bar_colors = top_df["Modality"].map({
    "spec": "steelblue",
    "shg": "darkorange"
})

plt.figure(figsize=(12, 6))
bars = plt.barh(top_df["Feature"], top_df["Importance"], color=bar_colors)

plt.xlabel("Mean abs back-projected LDA coefficient")
plt.title("Top 20 SHG + Spectral Features")

for bar, p_val, star in zip(bars, top_df["p_value"], top_df["significance"]):
    width = bar.get_width()
    text = star if star else f"{p_val:.2e}"
    offset = 0.02 * max(top_df["Importance"].max(), 1)

    plt.text(
        width + offset,
        bar.get_y() + bar.get_height() / 2,
        text,
        va="center",
        ha="left",
        fontsize=9
    )

legend_elements = [
    mpatches.Patch(color="steelblue", label="Spectral"),
    mpatches.Patch(color="darkorange", label="SHG"),
    mpatches.Patch(facecolor="none", edgecolor="none", label="* p<0.05"),
    mpatches.Patch(facecolor="none", edgecolor="none", label="** p<0.01"),
    mpatches.Patch(facecolor="none", edgecolor="none", label="*** p<0.001"),
]
plt.legend(handles=legend_elements, loc="lower right", frameon=False)

plt.tight_layout()
plt.savefig(os.path.join(save_path, f"{name}_Top20_features.png"), dpi=300, bbox_inches="tight")
plt.show()

top_df.to_csv(os.path.join(save_path, f"{name}_Top20_features.csv"), index=False)


# =========================================================
# Summary
# =========================================================
summary_df = pd.DataFrame({
    "n_samples": [len(combined_df)],
    "n_features_total": [X.shape[1]],
    "n_features_spec": [len(spec_cols)],
    "n_features_shg": [X.shape[1] - len(spec_cols)],
    "n_pca_components": [X_pca.shape[1]],
    "n_lda_components": [lda_n_components]
})
summary_df.to_csv(os.path.join(save_path, f"{name}_summary.csv"), index=False)

print("\nDone.")
print(f"Saved to: {save_path}")
print("Main outputs:")
print(f"  {name}_PCA_LDA_scatter.png")
print(f"  {name}_confusion_matrix.png")
print(f"  {name}_feature_importance.csv")
print(f"  {name}_Top20_features.png")