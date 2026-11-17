import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import f_classif


# =========================================================
# Settings
# =========================================================
folder_config = {
    "name": "replicate1",
    "base_path": "../res-15/04202026",
    "glcm_csv": "04202026_glcm_patch32_offset5_15slice.csv",
    "morph_csv": "04202026_patch32_features.csv"
}

name = "04202026"
save_path = "../res-15/single_replicate_pca_lda"
os.makedirs(save_path, exist_ok=True)

N_PER_GROUP = 256
RANDOM_STATE = 42
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
    label = re.sub(r'_\d+$', '', str(file_name))
    return label.upper()


def p_to_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


# =========================================================
# Load and merge
# =========================================================
glcm_path = os.path.join(folder_config["base_path"], folder_config["glcm_csv"])
morph_path = os.path.join(folder_config["base_path"], folder_config["morph_csv"])

glcm_df = pd.read_csv(glcm_path)
morph_df = pd.read_csv(morph_path)

print("glcm_df shape:", glcm_df.shape)
print("morph_df shape:", morph_df.shape)

merge_keys = ["file_name", "x", "y"]

duplicate_cols = [c for c in glcm_df.columns if c in morph_df.columns and c not in merge_keys]
print("Duplicate columns:", duplicate_cols)

# keep morphology version of duplicate columns
glcm_df_clean = glcm_df.drop(columns=duplicate_cols, errors="ignore")

merged_df = pd.merge(
    glcm_df_clean,
    morph_df,
    on=merge_keys,
    how="inner"
)

merged_df["label"] = merged_df["file_name"].apply(extract_label)

print("merged_df shape:", merged_df.shape)
print("Label counts:")
print(merged_df["label"].value_counts())


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
# Randomly sample same number per group
# =========================================================
sampled_groups = []

for group in label_order:
    group_df = merged_df[merged_df["label"] == group].copy()

    if len(group_df) == 0:
        continue

    if len(group_df) < N_PER_GROUP:
        raise ValueError(
            f"Group '{group}' has only {len(group_df)} rows, fewer than {N_PER_GROUP}."
        )

    sampled_group = group_df.sample(n=N_PER_GROUP, random_state=RANDOM_STATE)
    sampled_groups.append(sampled_group)

final_df = pd.concat(sampled_groups, axis=0).reset_index(drop=True)

print("\nAfter random sampling:")
print(final_df["label"].value_counts())
print("final_df shape:", final_df.shape)


# =========================================================
# Build feature matrix
# =========================================================
drop_cols = ["file_name", "x", "y", "label"]
feature_cols = [c for c in final_df.columns if c not in drop_cols]

print("Number of features:", len(feature_cols))

X = final_df[feature_cols].apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)

if X.isna().sum().sum() > 0:
    print("Found NaNs, filling with column median.")
    X = X.fillna(X.median(numeric_only=True))

y = final_df["label"].values


# =========================================================
# Standardize
# =========================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


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
    raise ValueError("Not enough classes for LDA.")

lda = LinearDiscriminantAnalysis(n_components=lda_n_components)
X_lda = lda.fit_transform(X_pca, y)

print("X_lda shape:", X_lda.shape)


# =========================================================
# LDA DataFrame
# =========================================================
lda_df = pd.DataFrame({
    "LDA1": X_lda[:, 0],
    "label": y
})

if lda_n_components > 1:
    lda_df["LDA2"] = X_lda[:, 1]
else:
    lda_df["LDA2"] = 0.0

lda_df.to_csv(os.path.join(save_path, f"{name}_projection.csv"), index=False)


# =========================================================
# Scatter plot
# =========================================================
plt.figure(figsize=(10, 8))

for group in label_order:
    mask = lda_df["label"] == group
    if np.sum(mask) == 0:
        continue

    plt.scatter(
        lda_df.loc[mask, "LDA1"],
        lda_df.loc[mask, "LDA2"],
        color=ratio_color_map.get(group, "gray"),
        s=50,
        alpha=0.75,
        edgecolors="none",
        label=group
    )

plt.xlabel("LDA Component 1", fontsize=16)
plt.ylabel("LDA Component 2", fontsize=16)
plt.title("PCA → LDA", fontsize=18)
plt.legend(title="Group", bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(save_path, f"{name}_scatter.png"), dpi=300, bbox_inches="tight")
plt.show()


# =========================================================
# Feature importance
# Back-project LDA coefficients to original feature space
# =========================================================
coef_pca_space = lda.coef_
coef_original_space = coef_pca_space @ pca.components_
importance = np.mean(np.abs(coef_original_space), axis=0)

importance_df = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

importance_df.to_csv(
    os.path.join(save_path, f"{name}_feature_importance.csv"),
    index=False
)

print("Feature importance saved.")


# =========================================================
# ANOVA
# =========================================================
F_values, p_values = f_classif(X_scaled, y)

feat_df = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": importance,
    "F_value": F_values,
    "p_value": p_values
}).sort_values(by="Importance", ascending=False)

feat_df.to_csv(
    os.path.join(save_path, f"{name}_all_features_importance_pvalues.csv"),
    index=False
)


# =========================================================
# Top 15 features
# =========================================================
top_df = feat_df.head(15).copy()
top_df["Importance_signed"] = top_df["Importance"]
top_df["significance"] = top_df["p_value"].apply(p_to_stars)
top_df = top_df.sort_values("Importance_signed")

plt.figure(figsize=(12, 5))
bars = plt.barh(top_df["Feature"], top_df["Importance_signed"], color="skyblue")
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
plt.savefig(os.path.join(save_path, f"{name}_Top15_pvalues.png"), dpi=300, bbox_inches="tight")
plt.show()

top_df.to_csv(
    os.path.join(save_path, f"{name}_Top15_features_with_pvalues.csv"),
    index=False
)


# =========================================================
# Save summary
# =========================================================
summary_df = pd.DataFrame({
    "n_samples": [len(final_df)],
    "n_features": [len(feature_cols)],
    "n_pca_components": [X_pca.shape[1]],
    "n_lda_components": [lda_n_components]
})
summary_df.to_csv(os.path.join(save_path, f"{name}_summary.csv"), index=False)

print("\nSaved:")
print(f"  -> {name}_projection.csv")
print(f"  -> {name}_scatter.png")
print(f"  -> {name}_feature_importance.csv")
print(f"  -> {name}_all_features_importance_pvalues.csv")
print(f"  -> {name}_Top15_features_with_pvalues.csv")
print(f"  -> {name}_Top15_pvalues.png")
print(f"  -> {name}_summary.csv")