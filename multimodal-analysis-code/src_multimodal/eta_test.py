# Minimal revision: original LDA weight calculation and feature selection retained.
# Removes ANOVA p-values, significance stars and their legend.
# NOTE: bars still show lda.coef_[0], not LD1/LD2 projection weights.
# Exploratory analysis: one independently prepared specimen per ratio.

import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import re

# =========================================================
# Settings
# =========================================================
name = '03252026'
path = '../res-15/03252026'
save_path = '../res-15/03252026/lda_res_32_5_v2_eta_squared'
os.makedirs(save_path, exist_ok=True)

shg_glcm_path = os.path.join(path, '03252026_glcm_patch32_offset5_15slice.csv')
shg_morph_path = os.path.join(path, '03252026_patch32_features_v2.csv')

N_PER_GROUP = 128
RANDOM_STATE = 42

# =========================================================
# Same color map as your spectral code
# =========================================================
def build_ratio_color_map():
    """
    Use the exact same tab20 color mapping.
    SER = dark color
    PSER = lighter paired color
    """
    tab20 = plt.get_cmap("tab20").colors

    ratio_color_map = {
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
    return ratio_color_map

ratio_color_map = build_ratio_color_map()

# =========================================================
# Load data
# =========================================================
glcm_df = pd.read_csv(shg_glcm_path)
morph_df = pd.read_csv(shg_morph_path)

print("glcm_df shape:", glcm_df.shape)
print("morph_df shape:", morph_df.shape)

# =========================================================
# Clean duplicate columns before merge
# =========================================================
merge_keys = ['file_name', 'x', 'y']

duplicate_cols = [c for c in glcm_df.columns if c in morph_df.columns and c not in merge_keys]
print("Duplicate columns:", duplicate_cols)

# keep morphology version of duplicate columns
glcm_df_clean = glcm_df.drop(columns=duplicate_cols, errors='ignore')

merged_df = pd.merge(
    glcm_df_clean,
    morph_df,
    on=merge_keys,
    how='inner'
)

print("merged_df shape:", merged_df.shape)

# =========================================================
# Create label from file_name
# lower/upper case treated the same
# Example:
#   1000_1       -> 1000
#   9010SER_2    -> 9010SER
#   6040PSER_10  -> 6040PSER
# =========================================================
def extract_label(file_name):
    label = re.sub(r'_\d+$', '', str(file_name))
    return label.upper()

merged_df['label'] = merged_df['file_name'].apply(extract_label)

print("Original label counts:")
print(merged_df['label'].value_counts())

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

existing_labels = merged_df['label'].unique().tolist()
label_order = [g for g in desired_order if g in existing_labels] + \
              [g for g in sorted(existing_labels) if g not in desired_order]

print("Final label order:", label_order)

# =========================================================
# Randomly select 256 from each group
# =========================================================
sampled_groups = []

for group in label_order:
    group_df = merged_df[merged_df['label'] == group].copy()

    if len(group_df) == 0:
        continue

    if len(group_df) < N_PER_GROUP:
        raise ValueError(
            f"Group '{group}' has only {len(group_df)} rows, fewer than {N_PER_GROUP}."
        )

    sampled_group = group_df.sample(n=N_PER_GROUP, random_state=RANDOM_STATE)
    sampled_groups.append(sampled_group)

final_df = pd.concat(sampled_groups, axis=0).reset_index(drop=True)

print("After random sampling:")
print(final_df['label'].value_counts())
print("final_df shape:", final_df.shape)

# =========================================================
# Build feature matrix
# =========================================================
drop_cols = ['file_name', 'x', 'y', 'label']
feature_cols = [c for c in final_df.columns if c not in drop_cols]

print("Number of features:", len(feature_cols))

# numeric only
X = final_df[feature_cols].apply(pd.to_numeric, errors='coerce')
X = X.replace([np.inf, -np.inf], np.nan)

if X.isna().sum().sum() > 0:
    print("Found NaNs, filling with column median.")
    X = X.fillna(X.median(numeric_only=True))

y = final_df['label'].values

# =========================================================
# Standardize features
# =========================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================================================
# LDA
# =========================================================
n_classes = len(np.unique(y))
n_components = min(2, n_classes - 1)

lda = LinearDiscriminantAnalysis(n_components=n_components)
X_lda = lda.fit_transform(X_scaled, y)

print(f"LDA shape: {X_lda.shape}")
print(f"Number of samples: {len(y)}")

# =========================================================
# Create LDA DataFrame
# =========================================================
lda_df = pd.DataFrame({
    "LDA1": X_lda[:, 0],
    "Group": y
})

if n_components > 1:
    lda_df["LDA2"] = X_lda[:, 1]
else:
    lda_df["LDA2"] = 0.0

print("lda_df shape:", lda_df.shape)
print(lda_df['Group'].value_counts())

# =========================================================
# LDA scatter plot using SAME colors
# =========================================================
plt.figure(figsize=(12, 8))

for group in label_order:
    mask = lda_df["Group"] == group
    if np.sum(mask) == 0:
        continue

    plt.scatter(
        lda_df.loc[mask, "LDA1"],
        lda_df.loc[mask, "LDA2"],
        label=group,
        color=ratio_color_map.get(group, "gray"),
        s=50,
        alpha=0.75,
        edgecolors="none"
    )

plt.xlabel('LDA Component 1',fontsize=20)
plt.ylabel('LDA Component 2',fontsize=20)
plt.xticks(fontsize=21)
plt.yticks(fontsize=21)
plt.title('PCA → LDA: SHG Features',fontsize=24)
plt.legend(title='Group', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(save_path, f'{name}_LDA_SHG.png'), dpi=300, bbox_inches='tight')
plt.show()

# =========================================================
# LDA feature importance
# =========================================================
feature_names = X.columns

if lda.coef_.ndim == 1:
    lda_importances = np.abs(lda.coef_)
else:
    lda_importances = np.mean(np.abs(lda.coef_), axis=0)

# Descriptive eta-squared for each feature across the measured specimens.
# Uses the same sampled, median-imputed features as LDA.
# This is not a test of reproducible composition effects.
def descriptive_eta_squared(values, labels):
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels)
    if values.ndim != 2 or labels.ndim != 1 or len(labels) != len(values):
        raise ValueError("Expected a 2D feature matrix and one label per row.")
    if not np.isfinite(values).all():
        raise ValueError("Eta-squared requires finite values after preprocessing.")
    grand_mean = values.mean(axis=0)
    ss_total = np.sum((values - grand_mean) ** 2, axis=0)
    ss_between = np.zeros(values.shape[1], dtype=float)
    for group in np.unique(labels):
        group_values = values[labels == group]
        ss_between += len(group_values) * (group_values.mean(axis=0) - grand_mean) ** 2
    eta_squared = np.divide(
        ss_between, ss_total,
        out=np.full(values.shape[1], np.nan), where=ss_total > 0
    )
    return np.clip(eta_squared, 0, 1)

eta_squared = descriptive_eta_squared(X_scaled, y)
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': lda_importances,
    'Eta_squared_descriptive': eta_squared
}).sort_values(by='Importance', ascending=False)

importance_df.to_csv(
    os.path.join(save_path, f'shg_lda_feature_importance_{name}.csv'),
    index=False
)
feat_df = importance_df.copy()
feat_df.to_csv(
    os.path.join(save_path, f'{name}_all_features_importance.csv'),
    index=False
)

# =========================================================
# Top 10 features
# =========================================================
top20_df = feat_df.head(10).copy()

if lda.coef_.ndim == 1:
    coef_for_sign = lda.coef_
else:
    coef_for_sign = lda.coef_[0]

coef_map = dict(zip(feature_names, coef_for_sign))
top20_df['Importance_signed'] = top20_df['Feature'].map(coef_map)

top20_df = top20_df.sort_values('Importance_signed')

# =========================================================
# Plot Top 10 signed feature importance
# =========================================================
colors = top20_df['Importance_signed'].apply(lambda x: 'skyblue' if x >= 0 else 'salmon')

plt.figure(figsize=(12, 8))
bars = plt.barh(top20_df['Feature'], top20_df['Importance_signed'], color=colors)
plt.axvline(0, color='black', linewidth=0.8)
plt.xlabel('LDA Coefficient')
plt.title('Top 10 LDA Feature Importance', fontsize=21)

# A separate right-hand annotation column, aligned with feature rows.
ax = plt.gca()
ax.text(1.03, 1.02, r'$\eta^2$ (descriptive)', transform=ax.transAxes,
        ha='left', va='bottom', fontsize=13)
for bar, eta in zip(bars, top20_df['Eta_squared_descriptive']):
    text = f"{eta:.3f}" if np.isfinite(eta) else "undefined"
    ax.text(1.03, bar.get_y() + bar.get_height() / 2, text,
            transform=ax.get_yaxis_transform(), ha='left', va='center', fontsize=16)
# plt.figtext(0.5, 0.01,
#             'One specimen per ratio; eta-squared describes sampled patch variation.',
#             ha='center', fontsize=10)


plt.xticks(fontsize=21)
plt.yticks(fontsize=21)

plt.tight_layout(rect=[0, 0.04, 0.82, 1])
plt.savefig(os.path.join(save_path, f'{name}_shg_LDA_Top10_eta_squared.png'), dpi=300, bbox_inches='tight')
plt.show()

# =========================================================
# Save outputs
# =========================================================
lda_df.to_csv(os.path.join(save_path, f'{name}_shg_LDA_projection.csv'), index=False)
top20_df.to_csv(os.path.join(save_path, f'{name}_shg_LDA_Top10_features.csv'), index=False)

print("Saved:")
print(f"  -> {name}_LDA_SHG.png")
print(f"  -> shg_lda_feature_importance_{name}.csv")
print(f"  -> {name}_all_features_importance.csv")
print(f"  -> {name}_shg_LDA_projection.csv")
print(f"  -> {name}_shg_LDA_Top10_features.csv")
print(f"  -> {name}_shg_LDA_Top10_eta_squared.png")