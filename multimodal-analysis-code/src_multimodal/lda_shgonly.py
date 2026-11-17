import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.feature_selection import f_classif
import numpy as np
import re
import matplotlib.patches as mpatches

# =========================================================
# Settings
# =========================================================
name = '03272026'
path = '../res-15/03272026'
save_path = '../res-15/03272026/lda_res_32_5_v2'
os.makedirs(save_path, exist_ok=True)

shg_glcm_path = os.path.join(path, '03272026_glcm_patch32_offset5_15slice.csv')
shg_morph_path = os.path.join(path, '03272026_patch32_features_v2.csv')

N_PER_GROUP = 256
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
# Randomly select 128 from each group
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

plt.xlabel('LDA Component 1',fontsize=16)
plt.ylabel('LDA Component 2',fontsize=16)
plt.xticks(fontsize=21)
plt.yticks(fontsize=21)
plt.title('LDA: SHG Features',fontsize=18)
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

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': lda_importances
}).sort_values(by='Importance', ascending=False)

importance_df.to_csv(
    os.path.join(save_path, f'shg_lda_feature_importance_{name}.csv'),
    index=False
)
print("LDA feature importance saved.")

# =========================================================
# ANOVA F-test + p-values
# =========================================================
F_values, p_values = f_classif(X_scaled, y)

feat_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': lda_importances,
    'F_value': F_values,
    'p_value': p_values
}).sort_values(by='Importance', ascending=False)

feat_df.to_csv(
    os.path.join(save_path, f'{name}_all_features_importance_pvalues.csv'),
    index=False
)

# =========================================================
# Top 20 features
# =========================================================
top20_df = feat_df.head(10).copy()

if lda.coef_.ndim == 1:
    coef_for_sign = lda.coef_
else:
    coef_for_sign = lda.coef_[0]

coef_map = dict(zip(feature_names, coef_for_sign))
top20_df['Importance_signed'] = top20_df['Feature'].map(coef_map)

def p_to_stars(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return ''

top20_df['significance'] = top20_df['p_value'].apply(p_to_stars)
top20_df = top20_df.sort_values('Importance_signed')

# =========================================================
# Plot Top 20 signed feature importance
# =========================================================
colors = top20_df['Importance_signed'].apply(lambda x: 'skyblue' if x >= 0 else 'salmon')

plt.figure(figsize=(12, 8))
bars = plt.barh(top20_df['Feature'], top20_df['Importance_signed'], color=colors)
plt.axvline(0, color='black', linewidth=0.8)
plt.xlabel('LDA Coefficient')
plt.title(f'Top 10 LDA Feature Importance',fontsize=21)

for bar, p_val, star in zip(bars, top20_df['p_value'], top20_df['significance']):
    width = bar.get_width()
    text = star if star else f"{p_val:.2e}"
    offset = 0.02 * max(abs(top20_df['Importance_signed'].max()), abs(top20_df['Importance_signed'].min()), 1)

    if width >= 0:
        plt.text(width + offset, bar.get_y() + bar.get_height()/2, text,
                 va='center', ha='left', fontsize=18)
    else:
        plt.text(width - offset, bar.get_y() + bar.get_height()/2, text,
                 va='center', ha='right', fontsize=18)

legend_elements = [
    mpatches.Patch(facecolor='none', edgecolor='none', label='* p<0.05'),
    mpatches.Patch(facecolor='none', edgecolor='none', label='** p<0.01'),
    mpatches.Patch(facecolor='none', edgecolor='none', label='*** p<0.001')
]
plt.legend(handles=legend_elements, loc='lower right', frameon=False , fontsize=18)
plt.xticks(fontsize=21)
plt.yticks(fontsize=21)

plt.tight_layout()
plt.savefig(os.path.join(save_path, f'{name}_shg_LDA_Top10_pvalues.png'), dpi=300, bbox_inches='tight')
plt.show()

# =========================================================
# Save outputs
# =========================================================
lda_df.to_csv(os.path.join(save_path, f'{name}_shg_LDA_projection.csv'), index=False)
top20_df.to_csv(os.path.join(save_path, f'{name}_shg_LDA_Top10_features_with_pvalues.csv'), index=False)

print("Saved:")
print(f"  -> {name}_LDA_SHG.png")
print(f"  -> shg_lda_feature_importance_{name}.csv")
print(f"  -> {name}_all_features_importance_pvalues.csv")
print(f"  -> {name}_shg_LDA_projection.csv")
print(f"  -> {name}_shg_LDA_Top20_features_with_pvalues.csv")
print(f"  -> {name}_shg_LDA_Top20_pvalues.png")