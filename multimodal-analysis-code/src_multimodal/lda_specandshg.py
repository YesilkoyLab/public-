import os
import re
import random
import numpy as np
import pandas as pd
import seaborn as sns
from glob import glob
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# =========================================================
# Paths / settings
# =========================================================
name = '03272026'
path = '../res/03272026/'
shg_glcm_path = path + '03272026_glcm_patch32_offset5.csv'
shg_morph_path = path + '03272026_patch32_features.csv'
spectral_root = r'/Volumes/TIANYI/SPEC/res/Caf2_03272026/2nd'

# case-insensitive now handled by extract_label + folder matching
ratios = [ '1000', '9010PSER', '8020PSER', '7030PSER', '6040PSER', '9010SER', '8020SER', '7030SER', '6040SER']
subfolder = 'LMT_1'
sample_per_group = 128

random.seed(42)
np.random.seed(42)

# =========================================================
# Same color mapping as your spectral code
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
# Helper functions
# =========================================================
def extract_label(file_name):
    """
    Examples:
      1000_1       -> 1000
      9010ser_2    -> 9010SER
      6040Pser_10  -> 6040PSER
    """
    label = re.sub(r'_\d+$', '', str(file_name))
    return label.upper()

def find_case_insensitive_subdir(parent_dir, target_name):
    """
    Find a folder under parent_dir whose name matches target_name ignoring case.
    """
    if not os.path.isdir(parent_dir):
        raise ValueError(f"Directory not found: {parent_dir}")

    target_upper = target_name.upper()
    for item in os.listdir(parent_dir):
        full_path = os.path.join(parent_dir, item)
        if os.path.isdir(full_path) and item.upper() == target_upper:
            return item

    raise ValueError(f"Cannot find folder matching '{target_name}' under: {parent_dir}")

# =========================================================
# Load SHG features (new structure)
# =========================================================
glcm_df = pd.read_csv(shg_glcm_path)
morph_df = pd.read_csv(shg_morph_path)

print("glcm_df shape:", glcm_df.shape)
print("morph_df shape:", morph_df.shape)

merge_keys = ['file_name', 'x', 'y']

# Remove duplicate feature columns before merge
duplicate_cols = [c for c in glcm_df.columns if c in morph_df.columns and c not in merge_keys]
print("Duplicate columns:", duplicate_cols)

# Keep morphology version for overlapping columns
glcm_df_clean = glcm_df.drop(columns=duplicate_cols, errors='ignore')

merged_df = pd.merge(glcm_df_clean, morph_df, on=merge_keys, how='inner')
merged_df['label'] = merged_df['file_name'].apply(extract_label)

print("merged_df shape:", merged_df.shape)
print("Merged SHG label counts:")
print(merged_df['label'].value_counts())

# Keep only requested ratios
ratio_set = {r.upper() for r in ratios}
shg_df = merged_df[merged_df['label'].isin(ratio_set)].copy().reset_index(drop=True)

# Shuffle to avoid any ordering bias before sampling
shg_df = shg_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Filtered shg_df shape:", shg_df.shape)
print("Filtered SHG label counts:")
print(shg_df['label'].value_counts())

# =========================================================
# Collect spectral data (unchanged logic, but folder matching is case-insensitive)
# =========================================================
spectral_data = []
labels = []

for ratio in ratios:
    matched_ratio_folder = find_case_insensitive_subdir(spectral_root, ratio)
    folder_path = os.path.join(spectral_root, matched_ratio_folder, subfolder)

    npy_files = sorted(glob(os.path.join(folder_path, '*.npy')))
    if len(npy_files) < sample_per_group:
        raise ValueError(
            f"Not enough spectral files for {ratio} in {folder_path}. "
            f"Found {len(npy_files)}, need {sample_per_group}."
        )

    sampled_files = random.sample(npy_files, sample_per_group)

    for f in sampled_files:
        spectrum = np.load(f)
        spectral_data.append(spectrum)
        labels.append(ratio.upper())

spectral_df = pd.DataFrame(spectral_data)
spectral_df['label'] = labels

# Normalize spectral data only
spectral_feat = spectral_df.drop(columns='label')
spectral_scaled = StandardScaler().fit_transform(spectral_feat)
spectral_df = pd.DataFrame(spectral_scaled, columns=spectral_feat.columns)
spectral_df['label'] = labels

# Rename spectral columns: 950 to 1800 step 2
new_column_names = list(range(950, 1802, 2))
old_column_names = spectral_df.columns[:426]
rename_dict = {old: new for old, new in zip(old_column_names, new_column_names)}
spectral_df.rename(columns=rename_dict, inplace=True)

print("spectral_df shape:", spectral_df.shape)
print("Spectral label counts:")
print(spectral_df['label'].value_counts())

# =========================================================
# Match SHG rows to spectral labels group-by-group
# =========================================================
shg_selected = []

for ratio in ratios:
    ratio_upper = ratio.upper()
    group_df = shg_df[shg_df['label'] == ratio_upper].copy()

    if len(group_df) < sample_per_group:
        raise ValueError(
            f"Not enough SHG patches for {ratio_upper}. "
            f"Found {len(group_df)}, need {sample_per_group}."
        )

    group_df = group_df.sample(n=sample_per_group, random_state=42).reset_index(drop=True)
    shg_selected.append(group_df)

shg_selected_df = pd.concat(shg_selected, axis=0, ignore_index=True)

# reorder exactly by ratios block order
ordered_shg_blocks = []
for ratio in ratios:
    ratio_upper = ratio.upper()
    ordered_shg_blocks.append(
        shg_selected_df[shg_selected_df['label'] == ratio_upper].reset_index(drop=True)
    )

shg_selected_df = pd.concat(ordered_shg_blocks, axis=0, ignore_index=True)

print("shg_selected_df shape:", shg_selected_df.shape)
print("Selected SHG label counts:")
print(shg_selected_df['label'].value_counts())

# =========================================================
# Combine spectral + SHG
# =========================================================
shg_feature_df = shg_selected_df.drop(columns=['file_name', 'x', 'y'], errors='ignore').copy()

# ensure labels align
if not np.array_equal(spectral_df['label'].values, shg_feature_df['label'].values):
    raise ValueError("Spectral labels and SHG labels do not align after sampling.")

combined_df = pd.concat(
    [
        spectral_df.reset_index(drop=True),
        shg_feature_df.drop(columns=['label']).reset_index(drop=True)
    ],
    axis=1
)

final_df = combined_df.copy()
print("Final shape:", final_df.shape)
print("Final label counts:")
print(final_df['label'].value_counts())

# =========================================================
# LDA preprocessing
# =========================================================
X = final_df.drop(columns=['label'])
y = final_df['label'].values

X.columns = X.columns.astype(str)
X = X.apply(pd.to_numeric, errors='coerce')
X = X.replace([np.inf, -np.inf], np.nan)

if X.isna().sum().sum() > 0:
    print("Found NaNs in X. Filling with column median.")
    X = X.fillna(X.median(numeric_only=True))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================================================
# Weighting
# =========================================================
num_spectral = 426
num_shg = X.shape[1] - num_spectral

# same weighting idea as your old script
weights = np.concatenate([np.full(num_spectral, 1), np.full(num_shg, 1)])
X_weighted = X_scaled * weights

# =========================================================
# LDA
# =========================================================
n_classes = len(np.unique(y))
n_components = min(2, n_classes - 1)

lda = LinearDiscriminantAnalysis(n_components=n_components)
X_lda = lda.fit_transform(X_weighted, y)

print(f"LDA shape: {X_lda.shape}")
print(f"Number of samples in y: {len(y)}")

# =========================================================
# Plot dataframe
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
# Plot
# =========================================================
palette = {g.upper(): ratio_color_map[g.upper()] for g in ratios if g.upper() in ratio_color_map}
ratios_upper = [r.upper() for r in ratios]

plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=lda_df,
    x="LDA1",
    y="LDA2",
    hue='Group',
    hue_order=ratios_upper,
    palette=palette,
    s=60,
    alpha=0.7
)
plt.xlabel('LDA Component 1')
plt.ylabel('LDA Component 2')
plt.title(f'LDA: SHG + Spectral ({name})')
plt.legend(title='Group', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(path, f'({name})_LDA_SHG+spec.png'), dpi=300, bbox_inches='tight')
plt.show()

# =========================================================
# Feature importance
# =========================================================
feature_names = X.columns

if lda.coef_.ndim == 1:
    lda_importances = np.abs(lda.coef_)
else:
    lda_importances = np.abs(lda.coef_).mean(axis=0)

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': lda_importances
}).sort_values(by='Importance', ascending=False)

importance_df.to_csv(
    os.path.join(path, f'shg+spec_lda_feature_importance_{name}.csv'),
    index=False
)


# =========================================================
# Feature importance heatmap
# =========================================================
coef_df = pd.DataFrame(lda.coef_, columns=X.columns)
coef_df.index = [f"{c}" for c in lda.classes_]

plt.figure(figsize=(14, 4))
sns.heatmap(
    coef_df,
    cmap='vlag',
    center=0,
    annot=False,
    cbar_kws={'label': 'LDA Coefficient'}
)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.title('Feature Importance Heatmap (LDA Coefficients)')
plt.tight_layout()
plt.savefig(os.path.join(path, 'shg+spec_lda_feature_importance_heatmap.png'), dpi=300, bbox_inches='tight')
plt.show()

print("\nDone.")
print(f"Saved to: {path}")
print(f"Main figure: ({name})_LDA_SHG+spec.png")