import os
import re
import random
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


# =========================================================
# Settings
# =========================================================
name = "03272026_multimodal_svm_rf"

# ---------- SHG ----------
shg_path = "../res/03272026/"
shg_glcm_path = os.path.join(shg_path, "03272026_glcm_patch32_offset5.csv")
shg_morph_path = os.path.join(shg_path, "03272026_patch32_features.csv")

# ---------- SPEC ----------
spectral_root = r"../../spec_analysis/res/Caf2_03252026/2nd"

# ---------- save ----------
save_path = os.path.join(shg_path, "svm_rf_multimodal_cv")
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

subfolders = ["LMT_1", "LMT_2", "LMT_3"]

# total matched samples per class
sample_per_group = 128

random_state = 42
random.seed(random_state)
np.random.seed(random_state)

n_splits = 5

# PCA options
use_pca = True
pca_n_components = 0.95   # or fixed int like 20

# Random forest
rf_n_estimators = 500


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


def find_case_insensitive_subdir(parent_dir, target_name):
    if not os.path.isdir(parent_dir):
        raise ValueError(f"Directory not found: {parent_dir}")

    target_upper = target_name.upper()
    for item in os.listdir(parent_dir):
        full_path = os.path.join(parent_dir, item)
        if os.path.isdir(full_path) and item.upper() == target_upper:
            return item

    raise ValueError(f"Cannot find folder matching '{target_name}' under: {parent_dir}")


def load_shg_features(glcm_csv, morph_csv, valid_labels):
    glcm_df = pd.read_csv(glcm_csv)
    morph_df = pd.read_csv(morph_csv)

    merge_keys = ["file_name", "x", "y"]
    duplicate_cols = [c for c in glcm_df.columns if c in morph_df.columns and c not in merge_keys]
    glcm_df_clean = glcm_df.drop(columns=duplicate_cols, errors="ignore")

    merged_df = pd.merge(glcm_df_clean, morph_df, on=merge_keys, how="inner")
    merged_df["label"] = merged_df["file_name"].apply(extract_label)
    merged_df = merged_df[merged_df["label"].isin(valid_labels)].copy()

    return merged_df


def load_spec_features(spectral_root, ratios, subfolders, sample_per_group, expected_length=426):
    """
    Sample spectra across all LMT folders, merged into one class per ratio.
    """
    spectral_rows = []

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
                raise ValueError(f"{f} length mismatch: expected {expected_length}, got {len(spectrum)}")

            spectral_rows.append({
                "file": f,
                "label": ratio.upper(),
                "subfolder": subfolder,
                "spectrum": spectrum
            })

    spectral_df = pd.DataFrame(spectral_rows)

    spec_mat = np.vstack(spectral_df["spectrum"].values)
    spec_feature_df = pd.DataFrame(spec_mat)
    spec_feature_df.columns = [str(w) for w in range(950, 1802, 2)]
    spec_feature_df["label"] = spectral_df["label"].values

    return spec_feature_df


def prepare_matched_datasets(shg_df, spec_df, ratios_upper, sample_per_group):
    """
    Match SHG and spectral data by class count only.
    Same number sampled from each class, then concatenate in same class order.
    """
    # -------- SHG --------
    shg_blocks = []
    for ratio in ratios_upper:
        group_df = shg_df[shg_df["label"] == ratio].copy()

        if len(group_df) < sample_per_group:
            raise ValueError(
                f"Not enough SHG samples for {ratio}. Found {len(group_df)}, need {sample_per_group}."
            )

        sampled = group_df.sample(n=sample_per_group, random_state=random_state).reset_index(drop=True)
        shg_blocks.append(sampled)

    shg_selected_df = pd.concat(shg_blocks, axis=0, ignore_index=True)

    ordered_shg_blocks = []
    for ratio in ratios_upper:
        ordered_shg_blocks.append(
            shg_selected_df[shg_selected_df["label"] == ratio].reset_index(drop=True)
        )
    shg_selected_df = pd.concat(ordered_shg_blocks, axis=0, ignore_index=True)

    # -------- SPEC --------
    ordered_spec_blocks = []
    for ratio in ratios_upper:
        group_df = spec_df[spec_df["label"] == ratio].copy()

        if len(group_df) < sample_per_group:
            raise ValueError(
                f"Not enough spectral samples for {ratio}. Found {len(group_df)}, need {sample_per_group}."
            )

        group_df = group_df.sample(n=sample_per_group, random_state=random_state).reset_index(drop=True)
        ordered_spec_blocks.append(group_df)

    spec_selected_df = pd.concat(ordered_spec_blocks, axis=0, ignore_index=True)

    # labels must align
    if not np.array_equal(spec_selected_df["label"].values, shg_selected_df["label"].values):
        raise ValueError("SHG labels and spectral labels do not align after sampling.")

    # SHG features only
    shg_feature_df = shg_selected_df.drop(columns=["file_name", "x", "y"], errors="ignore").copy()

    spec_X = spec_selected_df.drop(columns=["label"]).reset_index(drop=True)
    shg_X = shg_feature_df.drop(columns=["label"]).reset_index(drop=True)

    y = spec_selected_df["label"].values

    fused_X = pd.concat([spec_X, shg_X], axis=1)

    return spec_X, shg_X, fused_X, y


def clean_numeric_df(df):
    df = df.copy()
    df.columns = df.columns.astype(str)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    if df.isna().sum().sum() > 0:
        df = df.fillna(df.median(numeric_only=True))
    return df


def run_stratified_cv_model(X, y, model_name, modality_name, save_path,
                            use_pca=True, pca_n_components=0.95,
                            random_state=42, n_splits=5):
    """
    Train/evaluate one classifier on one modality with StratifiedKFold.
    Standardization and PCA are fit inside each fold.
    """
    X = clean_numeric_df(X)
    y = np.asarray(y)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    all_true = []
    all_pred = []
    fold_rows = []
    importances = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train = X.iloc[train_idx].values
        X_test = X.iloc[test_idx].values
        y_train = y[train_idx]
        y_test = y[test_idx]

        # ---------- standardize ----------
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # ---------- PCA ----------
        if use_pca:
            pca = PCA(n_components=pca_n_components, random_state=random_state)
            X_train_proc = pca.fit_transform(X_train_scaled)
            X_test_proc = pca.transform(X_test_scaled)
        else:
            pca = None
            X_train_proc = X_train_scaled
            X_test_proc = X_test_scaled

        # ---------- model ----------
        if model_name == "svm":
            model = SVC(kernel="rbf", C=1.0, gamma="scale")
        elif model_name == "rf":
            model = RandomForestClassifier(
                n_estimators=rf_n_estimators,
                random_state=random_state,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        model.fit(X_train_proc, y_train)
        y_pred = model.predict(X_test_proc)

        acc = accuracy_score(y_test, y_pred)

        fold_rows.append({
            "modality": modality_name,
            "model": model_name,
            "fold": fold,
            "accuracy": acc,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "n_features_input": X.shape[1],
            "n_features_after_pca": X_train_proc.shape[1]
        })

        all_true.extend(y_test)
        all_pred.extend(y_pred)

        # feature importance
        if model_name == "svm":
            # permutation is better, but here use linear proxy not available for RBF
            # save NaN placeholder
            importances.append(np.full(X.shape[1], np.nan))
        elif model_name == "rf":
            if pca is not None:
                # RF importance is in PCA space; back-project approximately
                imp = model.feature_importances_ @ np.abs(pca.components_)
            else:
                imp = model.feature_importances_
            importances.append(imp)

    # ---------- overall metrics ----------
    overall_acc = accuracy_score(all_true, all_pred)
    cm = confusion_matrix(all_true, all_pred, labels=ratios_upper)

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(
        os.path.join(save_path, f"{modality_name}_{model_name}_fold_results.csv"),
        index=False
    )

    # confusion matrix
    cm_df = pd.DataFrame(cm, index=ratios_upper, columns=ratios_upper)
    cm_df.to_csv(
        os.path.join(save_path, f"{modality_name}_{model_name}_confusion_matrix.csv")
    )

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ratios_upper)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    plt.title(f"{modality_name.upper()} + {model_name.upper()}\nAccuracy = {overall_acc:.3f}")
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_path, f"{modality_name}_{model_name}_confusion_matrix.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # feature importance (RF only meaningful here)
    imp_arr = np.vstack(importances)
    mean_imp = np.nanmean(imp_arr, axis=0)

    importance_df = pd.DataFrame({
        "Feature": X.columns.astype(str),
        "Importance": mean_imp
    }).sort_values(by="Importance", ascending=False)

    importance_df.to_csv(
        os.path.join(save_path, f"{modality_name}_{model_name}_feature_importance.csv"),
        index=False
    )

    return {
        "modality": modality_name,
        "model": model_name,
        "overall_accuracy": overall_acc,
        "mean_fold_accuracy": fold_df["accuracy"].mean(),
        "std_fold_accuracy": fold_df["accuracy"].std(ddof=1),
        "n_features_input": X.shape[1]
    }


# =========================================================
# Load data
# =========================================================
print("Loading SHG...")
shg_df = load_shg_features(shg_glcm_path, shg_morph_path, set(ratios_upper))
print("SHG loaded:", shg_df.shape)

print("Loading spectral...")
spec_df = load_spec_features(
    spectral_root=spectral_root,
    ratios=ratios,
    subfolders=subfolders,
    sample_per_group=sample_per_group
)
print("Spectral loaded:", spec_df.shape)

# =========================================================
# Match datasets
# =========================================================
spec_X, shg_X, fused_X, y = prepare_matched_datasets(
    shg_df=shg_df,
    spec_df=spec_df,
    ratios_upper=ratios_upper,
    sample_per_group=sample_per_group
)

print("spec_X shape:", spec_X.shape)
print("shg_X shape:", shg_X.shape)
print("fused_X shape:", fused_X.shape)
print("y shape:", y.shape)

# save matched labels
pd.DataFrame({"label": y}).to_csv(os.path.join(save_path, "matched_labels.csv"), index=False)

# =========================================================
# Run all models
# =========================================================
summary_rows = []

for modality_name, X_mod in [
    ("spec_only", spec_X),
    ("shg_only", shg_X),
    ("fused", fused_X),
]:
    for model_name in ["svm", "rf"]:
        print(f"\nRunning {model_name.upper()} on {modality_name} ...")
        result = run_stratified_cv_model(
            X=X_mod,
            y=y,
            model_name=model_name,
            modality_name=modality_name,
            save_path=save_path,
            use_pca=use_pca,
            pca_n_components=pca_n_components,
            random_state=random_state,
            n_splits=n_splits
        )
        summary_rows.append(result)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(save_path, "model_summary.csv"), index=False)

print("\nDone.")
print(f"Saved to: {save_path}")
print(summary_df)