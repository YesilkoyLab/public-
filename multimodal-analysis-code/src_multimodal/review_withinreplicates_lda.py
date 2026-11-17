import os
import re
import random
from glob import glob
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# =========================================================
# SETTINGS
# =========================================================

name = "biological_replicate_LOSO_LDA_within_replicate_zscore"

random_state = 42
random.seed(random_state)
np.random.seed(random_state)


# =========================================================
# CLASSES
# =========================================================

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


# =========================================================
# MIRSI TECHNICAL FOLDERS / ROIs
# =========================================================

subfolders = [
    "LMT_1",
    "LMT_2",
    # "LMT_3",
]


# =========================================================
# NUMBER OF TECHNICAL OBSERVATIONS PER CLASS
# PER BIOLOGICAL REPLICATE
# =========================================================

sample_per_group = 128


# =========================================================
# PCA
# =========================================================

use_pca = True
pca_n_components = 0.95


# =========================================================
# BIOLOGICAL REPLICATE PATHS
# =========================================================

replicates = {

    "REP1": {
        "shg_glcm":
            "../res/03252026/03252026_glcm_patch32_offset5.csv",

        "shg_morph":
            "../res/03252026/03252026_patch32_features.csv",

        "spectral_root":
            r"../../spec_analysis/res/Caf2_03252026/2nd",
    },

    "REP2": {
        "shg_glcm":
            "../res/03272026/03272026_glcm_patch32_offset5.csv",

        "shg_morph":
            "../res/03272026/03272026_patch32_features.csv",

        "spectral_root":
            r"../../spec_analysis/res/Caf2_03272026/2nd",
    },

    "REP3": {
        "shg_glcm":
            "../res/03272026/03272026_glcm_patch32_offset5.csv",

        "shg_morph":
            "../res/03272026/03272026_patch32_features.csv",

        "spectral_root":
            r"../../spec_analysis/res/Caf2_03272026/2nd",
    },
}


# =========================================================
# SAVE PATH
# =========================================================

save_path = "../res/biological_replicate_LOSO_LDA_within_replicate_zscore"

os.makedirs(save_path, exist_ok=True)


# =========================================================
# HELPERS
# =========================================================

def extract_label(file_name):

    label = re.sub(
        r"_\d+$",
        "",
        str(file_name),
    )

    return label.upper()


def find_case_insensitive_subdir(
    parent_dir,
    target_name,
):

    if not os.path.isdir(parent_dir):
        raise ValueError(
            f"Directory not found:\n{parent_dir}"
        )

    target_upper = target_name.upper()

    for item in os.listdir(parent_dir):

        full_path = os.path.join(
            parent_dir,
            item,
        )

        if (
            os.path.isdir(full_path)
            and item.upper() == target_upper
        ):
            return item

    raise ValueError(
        f"Cannot find folder matching "
        f"'{target_name}' under:\n"
        f"{parent_dir}"
    )


# =========================================================
# LOAD SHG FEATURES
# =========================================================

def load_shg_features(
    glcm_csv,
    morph_csv,
    valid_labels,
):

    print("\nLoading SHG")
    print("GLCM:", glcm_csv)
    print("Morph:", morph_csv)

    glcm_df = pd.read_csv(
        glcm_csv
    )

    morph_df = pd.read_csv(
        morph_csv
    )

    print(
        "GLCM shape:",
        glcm_df.shape
    )

    print(
        "Morphology shape:",
        morph_df.shape
    )

    merge_keys = [
        "file_name",
        "x",
        "y",
    ]

    duplicate_cols = [
        c
        for c in glcm_df.columns
        if c in morph_df.columns
        and c not in merge_keys
    ]

    glcm_clean = glcm_df.drop(
        columns=duplicate_cols,
        errors="ignore",
    )

    merged = pd.merge(
        glcm_clean,
        morph_df,
        on=merge_keys,
        how="inner",
    )

    merged["label"] = (
        merged["file_name"]
        .apply(extract_label)
    )

    merged = merged[
        merged["label"].isin(
            valid_labels
        )
    ].copy()

    print(
        "Merged SHG shape:",
        merged.shape
    )

    print(
        "\nSHG counts per class:"
    )

    print(
        merged["label"]
        .value_counts()
        .sort_index()
    )

    return merged


# =========================================================
# LOAD MIRSI FEATURES
# =========================================================

def load_spec_features(
    spectral_root,
    ratios,
    subfolders,
    sample_per_group,
    expected_length=426,
    random_state=42,
):

    rng = random.Random(
        random_state
    )

    spectral_rows = []

    per_folder = (
        sample_per_group
        // len(subfolders)
    )

    remainder = (
        sample_per_group
        % len(subfolders)
    )

    print("\nLoading MIRSI")
    print(spectral_root)

    for ratio in ratios:

        matched_ratio_folder = (
            find_case_insensitive_subdir(
                spectral_root,
                ratio,
            )
        )

        selected_files = []

        for i, subfolder in enumerate(
            subfolders
        ):

            folder_path = os.path.join(
                spectral_root,
                matched_ratio_folder,
                subfolder,
            )

            npy_files = sorted(
                glob(
                    os.path.join(
                        folder_path,
                        "*.npy",
                    )
                )
            )

            need_n = (
                per_folder
                + (
                    1
                    if i < remainder
                    else 0
                )
            )

            if len(npy_files) < need_n:

                raise ValueError(
                    "\nNot enough MIRSI files.\n"
                    f"Ratio: {ratio}\n"
                    f"Folder: {folder_path}\n"
                    f"Found: {len(npy_files)}\n"
                    f"Need: {need_n}"
                )

            sampled_files = rng.sample(
                npy_files,
                need_n,
            )

            for f in sampled_files:

                selected_files.append(
                    (
                        f,
                        subfolder,
                    )
                )

        for f, subfolder in selected_files:

            spectrum = np.load(
                f
            )

            spectrum = (
                np.asarray(
                    spectrum
                )
                .flatten()
            )

            if len(spectrum) != expected_length:

                raise ValueError(
                    f"Spectrum length mismatch:\n"
                    f"{f}\n"
                    f"Expected {expected_length}, "
                    f"got {len(spectrum)}"
                )

            if np.any(
                ~np.isfinite(
                    spectrum
                )
            ):

                raise ValueError(
                    f"Invalid spectral values:\n{f}"
                )

            spectral_rows.append({
                "file":
                    f,

                "label":
                    ratio.upper(),

                "subfolder":
                    subfolder,

                "spectrum":
                    spectrum,
            })

    spectral_df = pd.DataFrame(
        spectral_rows
    )

    spec_mat = np.vstack(
        spectral_df[
            "spectrum"
        ].values
    )

    spec_feature_df = pd.DataFrame(
        spec_mat
    )

    spec_feature_df.columns = [
        str(w)
        for w in range(
            950,
            1802,
            2,
        )
    ]

    spec_feature_df[
        "label"
    ] = spectral_df[
        "label"
    ].values

    print(
        "\nMIRSI counts per class:"
    )

    print(
        spec_feature_df[
            "label"
        ]
        .value_counts()
        .sort_index()
    )

    return spec_feature_df


# =========================================================
# NUMERIC CONVERSION
# =========================================================

def convert_numeric(
    df
):

    df = df.copy()

    df.columns = (
        df.columns
        .astype(str)
    )

    df = df.apply(
        pd.to_numeric,
        errors="coerce",
    )

    df = df.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    return df


# =========================================================
# ANALYSIS C:
# WITHIN-BIOLOGICAL-REPLICATE Z-SCORE
# =========================================================

def within_replicate_zscore(
    X,
):

    """
    Standardize each feature within ONE biological replicate:

        z = (x - mean_replicate) / std_replicate

    Mean and SD are calculated using ALL 9 formulations
    and their technical observations within that replicate.

    This removes replicate-specific offset and scale.

    NOTE:
    This is a batch-correction / sensitivity analysis,
    not a fully independent prediction normalization.
    """

    X = convert_numeric(
        X
    )

    replicate_mean = (
        X.mean(
            axis=0
        )
    )

    replicate_std = (
        X.std(
            axis=0,
            ddof=1,
        )
    )

    # Prevent division by zero
    zero_std_cols = (
        replicate_std[
            replicate_std == 0
        ]
        .index
    )

    if len(
        zero_std_cols
    ) > 0:

        print(
            "\nWarning: zero-variance features found:"
        )

        print(
            list(
                zero_std_cols
            )
        )

        replicate_std.loc[
            zero_std_cols
        ] = 1.0

    # Also protect against NaN SD
    bad_std_cols = (
        replicate_std[
            replicate_std.isna()
        ]
        .index
    )

    if len(
        bad_std_cols
    ) > 0:

        print(
            "\nWarning: NaN SD features found:"
        )

        print(
            list(
                bad_std_cols
            )
        )

        replicate_std.loc[
            bad_std_cols
        ] = 1.0

    X_z = (
        X - replicate_mean
    ) / replicate_std

    return (
        X_z,
        replicate_mean,
        replicate_std,
    )


# =========================================================
# PREPARE ONE BIOLOGICAL REPLICATE
# =========================================================

def prepare_one_replicate(
    replicate_name,
    config,
    ratios_upper,
    ratios,
    subfolders,
    sample_per_group,
    random_state,
):

    print("\n")
    print("=" * 75)
    print(
        f"PREPARING: {replicate_name}"
    )
    print("=" * 75)

    # -----------------------------------------------------
    # SHG
    # -----------------------------------------------------

    shg_df = load_shg_features(
        glcm_csv=
            config[
                "shg_glcm"
            ],

        morph_csv=
            config[
                "shg_morph"
            ],

        valid_labels=
            set(
                ratios_upper
            ),
    )


    # -----------------------------------------------------
    # MIRSI
    # -----------------------------------------------------

    spec_df = load_spec_features(
        spectral_root=
            config[
                "spectral_root"
            ],

        ratios=
            ratios,

        subfolders=
            subfolders,

        sample_per_group=
            sample_per_group,

        expected_length=
            426,

        random_state=
            random_state,
    )


    # =====================================================
    # BALANCED SHG SAMPLING
    # =====================================================

    shg_blocks = []

    for ratio in ratios_upper:

        group_df = shg_df[
            shg_df[
                "label"
            ] == ratio
        ].copy()

        if len(
            group_df
        ) < sample_per_group:

            raise ValueError(
                "\nNot enough SHG samples.\n"
                f"Replicate: {replicate_name}\n"
                f"Class: {ratio}\n"
                f"Found: {len(group_df)}\n"
                f"Need: {sample_per_group}"
            )

        sampled = (
            group_df
            .sample(
                n=
                    sample_per_group,

                random_state=
                    random_state,
            )
            .reset_index(
                drop=True
            )
        )

        shg_blocks.append(
            sampled
        )

    shg_selected = pd.concat(
        shg_blocks,
        axis=0,
        ignore_index=True,
    )


    # =====================================================
    # BALANCED MIRSI SAMPLING
    # =====================================================

    spec_blocks = []

    for ratio in ratios_upper:

        group_df = spec_df[
            spec_df[
                "label"
            ] == ratio
        ].copy()

        if len(
            group_df
        ) < sample_per_group:

            raise ValueError(
                "\nNot enough MIRSI samples.\n"
                f"Replicate: {replicate_name}\n"
                f"Class: {ratio}\n"
                f"Found: {len(group_df)}\n"
                f"Need: {sample_per_group}"
            )

        sampled = (
            group_df
            .sample(
                n=
                    sample_per_group,

                random_state=
                    random_state,
            )
            .reset_index(
                drop=True
            )
        )

        spec_blocks.append(
            sampled
        )

    spec_selected = pd.concat(
        spec_blocks,
        axis=0,
        ignore_index=True,
    )


    # =====================================================
    # VERIFY LABELS
    # =====================================================

    if not np.array_equal(
        shg_selected[
            "label"
        ].values,

        spec_selected[
            "label"
        ].values,
    ):

        raise ValueError(
            "SHG and MIRSI labels "
            "do not align."
        )


    # =====================================================
    # LABELS
    # =====================================================

    y = (
        spec_selected[
            "label"
        ].values
    )


    # =====================================================
    # SHG RAW FEATURES
    # =====================================================

    shg_X_raw = (
        shg_selected
        .drop(
            columns=[
                "file_name",
                "x",
                "y",
                "label",
            ],
            errors="ignore",
        )
        .copy()
    )

    shg_X_raw.columns = [
        f"SHG__{c}"
        for c in shg_X_raw.columns
    ]


    # =====================================================
    # MIRSI RAW FEATURES
    # =====================================================

    spec_X_raw = (
        spec_selected
        .drop(
            columns=[
                "label",
            ],
            errors="ignore",
        )
        .copy()
    )

    spec_X_raw.columns = [
        f"SPEC__{c}"
        for c in spec_X_raw.columns
    ]


    # =====================================================
    # WITHIN-REPLICATE Z-SCORE: SHG
    # =====================================================

    (
        shg_X_z,
        shg_mean,
        shg_std,
    ) = within_replicate_zscore(
        shg_X_raw
    )


    # =====================================================
    # WITHIN-REPLICATE Z-SCORE: MIRSI
    # =====================================================

    (
        spec_X_z,
        spec_mean,
        spec_std,
    ) = within_replicate_zscore(
        spec_X_raw
    )


    # =====================================================
    # SAVE NORMALIZATION PARAMETERS
    # =====================================================

    norm_dir = os.path.join(
        save_path,
        "within_replicate_normalization",
    )

    os.makedirs(
        norm_dir,
        exist_ok=True,
    )


    shg_norm_params = pd.DataFrame({
        "mean":
            shg_mean,

        "std":
            shg_std,
    })

    shg_norm_params.to_csv(
        os.path.join(
            norm_dir,
            f"{replicate_name}_"
            f"SHG_zscore_parameters.csv",
        )
    )


    spec_norm_params = pd.DataFrame({
        "mean":
            spec_mean,

        "std":
            spec_std,
    })

    spec_norm_params.to_csv(
        os.path.join(
            norm_dir,
            f"{replicate_name}_"
            f"MIRSI_zscore_parameters.csv",
        )
    )


    # =====================================================
    # SAVE NORMALIZED FEATURES
    # =====================================================

    normalized_dir = os.path.join(
        save_path,
        "normalized_features",
    )

    os.makedirs(
        normalized_dir,
        exist_ok=True,
    )


    shg_save = (
        shg_X_z.copy()
    )

    shg_save.insert(
        0,
        "label",
        y,
    )

    shg_save.insert(
        0,
        "replicate",
        replicate_name,
    )

    shg_save.to_csv(
        os.path.join(
            normalized_dir,
            f"{replicate_name}_"
            f"SHG_zscore.csv",
        ),
        index=False,
    )


    spec_save = (
        spec_X_z.copy()
    )

    spec_save.insert(
        0,
        "label",
        y,
    )

    spec_save.insert(
        0,
        "replicate",
        replicate_name,
    )

    spec_save.to_csv(
        os.path.join(
            normalized_dir,
            f"{replicate_name}_"
            f"MIRSI_zscore.csv",
        ),
        index=False,
    )


    # =====================================================
    # FUSED DATA
    # =====================================================

    fused_X_z = pd.concat(
        [
            spec_X_z.reset_index(
                drop=True
            ),

            shg_X_z.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )


    # =====================================================
    # GROUP ID
    # =====================================================

    groups = np.array(
        [
            replicate_name
        ] * len(y)
    )


    # =====================================================
    # SANITY CHECK
    # =====================================================

    print(
        f"\n{replicate_name} shapes"
    )

    print(
        "SHG:",
        shg_X_z.shape
    )

    print(
        "MIRSI:",
        spec_X_z.shape
    )

    print(
        "Fused:",
        fused_X_z.shape
    )


    print(
        "\nSHG mean after replicate z-score:"
    )

    print(
        np.nanmean(
            np.abs(
                shg_X_z.mean(
                    axis=0
                ).values
            )
        )
    )

    print(
        "SHG std after replicate z-score:"
    )

    print(
        np.nanmean(
            shg_X_z.std(
                axis=0,
                ddof=1,
            ).values
        )
    )


    print(
        "\nMIRSI mean after replicate z-score:"
    )

    print(
        np.nanmean(
            np.abs(
                spec_X_z.mean(
                    axis=0
                ).values
            )
        )
    )

    print(
        "MIRSI std after replicate z-score:"
    )

    print(
        np.nanmean(
            spec_X_z.std(
                axis=0,
                ddof=1,
            ).values
        )
    )


    return {

        "shg_X":
            shg_X_z,

        "spec_X":
            spec_X_z,

        "fused_X":
            fused_X_z,

        "y":
            y,

        "groups":
            groups,
    }


# =========================================================
# TRAINING-SET MEDIAN IMPUTATION
# =========================================================

def fill_nan_from_training(
    X_train,
    X_test,
):

    train_median = (
        X_train.median(
            axis=0
        )
    )

    X_train = (
        X_train.fillna(
            train_median
        )
    )

    X_test = (
        X_test.fillna(
            train_median
        )
    )

    bad_cols = (
        X_train.columns[
            X_train.isna().any()
        ]
    )

    if len(
        bad_cols
    ) > 0:

        print(
            "\nDropping problematic columns:"
        )

        print(
            list(
                bad_cols
            )
        )

        X_train = (
            X_train.drop(
                columns=
                    bad_cols
            )
        )

        X_test = (
            X_test.drop(
                columns=
                    bad_cols
            )
        )

    return (
        X_train,
        X_test,
    )


# =========================================================
# MAJORITY VOTE
# =========================================================

def majority_vote(
    predictions,
):

    return Counter(
        list(
            predictions
        )
    ).most_common(
        1
    )[0][0]


# =========================================================
# LDA FEATURE IMPORTANCE
# =========================================================

def backproject_lda_importance(
    model,
    pca,
    feature_names,
):

    coef_pca = (
        model.coef_
    )

    if coef_pca.ndim == 1:

        coef_pca = (
            coef_pca[
                None,
                :
            ]
        )

    if pca is not None:

        coef_original = (
            coef_pca
            @ pca.components_
        )

    else:

        coef_original = (
            coef_pca
        )

    importance = np.mean(
        np.abs(
            coef_original
        ),
        axis=0,
    )

    return pd.Series(
        importance,
        index=
            feature_names,
    )


# =========================================================
# LEAVE-ONE-BIOLOGICAL-REPLICATE-OUT LDA
# =========================================================

def run_logo_lda(
    X,
    y,
    groups,
    modality_name,
    save_path,
    ratios_upper,
    use_pca=True,
    pca_n_components=0.95,
):

    print("\n")
    print("=" * 75)
    print(
        f"WITHIN-REPLICATE Z-SCORE LOSO-LDA: "
        f"{modality_name}"
    )
    print("=" * 75)


    X = convert_numeric(
        X
    )

    y = np.asarray(
        y
    )

    groups = np.asarray(
        groups
    )


    logo = LeaveOneGroupOut()


    all_true = []
    all_pred = []

    fold_rows = []
    prediction_rows = []
    gel_vote_rows = []

    importance_list = []


    # =====================================================
    # CROSS-VALIDATION
    # =====================================================

    for fold, (
        train_idx,
        test_idx,
    ) in enumerate(

        logo.split(
            X,
            y,
            groups=
                groups,
        ),

        start=1,
    ):

        train_reps = np.unique(
            groups[
                train_idx
            ]
        )

        test_reps = np.unique(
            groups[
                test_idx
            ]
        )


        test_rep = (
            test_reps[0]
        )


        print("\n")
        print("-" * 65)

        print(
            f"FOLD {fold}"
        )

        print(
            "Train:",
            train_reps
        )

        print(
            "Test:",
            test_rep
        )


        # =================================================
        # TRAIN / TEST
        # =================================================

        X_train = (
            X.iloc[
                train_idx
            ]
            .copy()
        )

        X_test = (
            X.iloc[
                test_idx
            ]
            .copy()
        )

        y_train = (
            y[
                train_idx
            ]
        )

        y_test = (
            y[
                test_idx
            ]
        )


        # =================================================
        # MISSING VALUES
        # =================================================

        (
            X_train,
            X_test,
        ) = fill_nan_from_training(
            X_train,
            X_test,
        )

        feature_names = (
            X_train.columns
        )


        # =================================================
        # TRAINING-FOLD STANDARD SCALER
        #
        # This is technically redundant after within-
        # replicate z-scoring, but keeping it ensures
        # identical modeling workflow across analyses.
        #
        # It is fitted ONLY on training replicates.
        # =================================================

        scaler = (
            StandardScaler()
        )

        X_train_scaled = (
            scaler.fit_transform(
                X_train.values
            )
        )

        X_test_scaled = (
            scaler.transform(
                X_test.values
            )
        )


        # =================================================
        # PCA
        # =================================================

        if use_pca:

            pca = PCA(
                n_components=
                    pca_n_components
            )

            X_train_proc = (
                pca.fit_transform(
                    X_train_scaled
                )
            )

            X_test_proc = (
                pca.transform(
                    X_test_scaled
                )
            )

            print(
                "PCA components:",
                X_train_proc.shape[1]
            )

            print(
                "Explained variance:",
                round(
                    pca
                    .explained_variance_ratio_
                    .sum(),
                    4,
                ),
            )

        else:

            pca = None

            X_train_proc = (
                X_train_scaled
            )

            X_test_proc = (
                X_test_scaled
            )


        # =================================================
        # LDA
        # =================================================

        lda = (
            LinearDiscriminantAnalysis()
        )

        lda.fit(
            X_train_proc,
            y_train,
        )

        y_pred = (
            lda.predict(
                X_test_proc
            )
        )


        # =================================================
        # TECHNICAL ACCURACY
        # =================================================

        fold_technical_acc = (
            accuracy_score(
                y_test,
                y_pred,
            )
        )

        print(
            "Technical accuracy:",
            round(
                fold_technical_acc,
                4,
            ),
        )


        # =================================================
        # STORE TECHNICAL PREDICTIONS
        # =================================================

        for true_label, pred_label in zip(
            y_test,
            y_pred,
        ):

            prediction_rows.append({

                "fold":
                    fold,

                "replicate":
                    test_rep,

                "true_label":
                    true_label,

                "predicted_label":
                    pred_label,
            })


        all_true.extend(
            y_test
        )

        all_pred.extend(
            y_pred
        )


        # =================================================
        # GEL / FORMULATION MAJORITY VOTE
        # =================================================

        fold_pred_df = pd.DataFrame({

            "true":
                y_test,

            "pred":
                y_pred,
        })


        fold_gel_correct = []


        for true_class in ratios_upper:

            class_df = (
                fold_pred_df[
                    fold_pred_df[
                        "true"
                    ] == true_class
                ]
            )

            if len(
                class_df
            ) == 0:
                continue


            voted_prediction = (
                majority_vote(
                    class_df[
                        "pred"
                    ].values
                )
            )


            correct = int(
                voted_prediction
                == true_class
            )


            fold_gel_correct.append(
                correct
            )


            gel_vote_rows.append({

                "fold":
                    fold,

                "replicate":
                    test_rep,

                "true_label":
                    true_class,

                "predicted_label":
                    voted_prediction,

                "correct":
                    correct,

                "n_technical_observations":
                    len(
                        class_df
                    ),
            })


        fold_gel_accuracy = (
            np.mean(
                fold_gel_correct
            )
        )


        print(
            "Gel/formulation accuracy:",
            round(
                fold_gel_accuracy,
                4,
            ),
        )


        # =================================================
        # FOLD SUMMARY
        # =================================================

        fold_rows.append({

            "fold":
                fold,

            "modality":
                modality_name,

            "train_replicates":
                ",".join(
                    train_reps
                ),

            "test_replicate":
                test_rep,

            "technical_accuracy":
                fold_technical_acc,

            "gel_level_accuracy":
                fold_gel_accuracy,

            "n_train":
                len(
                    train_idx
                ),

            "n_test":
                len(
                    test_idx
                ),

            "n_features_input":
                X_train.shape[1],

            "n_features_after_pca":
                X_train_proc.shape[1],
        })


        # =================================================
        # FEATURE IMPORTANCE
        # =================================================

        importance = (
            backproject_lda_importance(

                model=
                    lda,

                pca=
                    pca,

                feature_names=
                    feature_names,
            )
        )


        importance.name = (
            f"fold_{fold}"
        )


        importance_list.append(
            importance
        )


    # =====================================================
    # SAVE RESULTS
    # =====================================================

    fold_df = pd.DataFrame(
        fold_rows
    )


    fold_df.to_csv(
        os.path.join(
            save_path,
            f"{modality_name}_"
            f"zscore_LOSO_fold_results.csv",
        ),
        index=False,
    )


    prediction_df = pd.DataFrame(
        prediction_rows
    )


    prediction_df.to_csv(
        os.path.join(
            save_path,
            f"{modality_name}_"
            f"zscore_technical_predictions.csv",
        ),
        index=False,
    )


    gel_vote_df = pd.DataFrame(
        gel_vote_rows
    )


    gel_vote_df.to_csv(
        os.path.join(
            save_path,
            f"{modality_name}_"
            f"zscore_gel_level_predictions.csv",
        ),
        index=False,
    )


    # =====================================================
    # TECHNICAL CONFUSION MATRIX
    # =====================================================

    overall_technical_accuracy = (
        accuracy_score(
            all_true,
            all_pred,
        )
    )


    technical_cm = (
        confusion_matrix(
            all_true,
            all_pred,
            labels=
                ratios_upper,
        )
    )


    pd.DataFrame(
        technical_cm,
        index=
            ratios_upper,
        columns=
            ratios_upper,
    ).to_csv(
        os.path.join(
            save_path,
            f"{modality_name}_"
            f"zscore_technical_confusion_matrix.csv",
        )
    )


    fig, ax = plt.subplots(
        figsize=(
            8,
            7,
        )
    )


    disp = ConfusionMatrixDisplay(
        confusion_matrix=
            technical_cm,

        display_labels=
            ratios_upper,
    )


    disp.plot(
        ax=ax,
        xticks_rotation=45,
        colorbar=False,
    )


    ax.set_title(
        f"{modality_name.upper()} "
        f"Z-SCORE LOSO-LDA\n"
        f"Technical accuracy = "
        f"{overall_technical_accuracy:.3f}"
    )


    plt.tight_layout()


    plt.savefig(
        os.path.join(
            save_path,
            f"{modality_name}_"
            f"zscore_technical_confusion_matrix.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )


    plt.close()


    # =====================================================
    # GEL-LEVEL CONFUSION MATRIX
    # =====================================================

    gel_true = (
        gel_vote_df[
            "true_label"
        ].values
    )


    gel_pred = (
        gel_vote_df[
            "predicted_label"
        ].values
    )


    overall_gel_accuracy = (
        accuracy_score(
            gel_true,
            gel_pred,
        )
    )


    gel_cm = (
        confusion_matrix(
            gel_true,
            gel_pred,
            labels=
                ratios_upper,
        )
    )


    pd.DataFrame(
        gel_cm,
        index=
            ratios_upper,
        columns=
            ratios_upper,
    ).to_csv(
        os.path.join(
            save_path,
            f"{modality_name}_"
            f"zscore_gel_confusion_matrix.csv",
        )
    )


    fig, ax = plt.subplots(
        figsize=(
            8,
            7,
        )
    )


    disp = ConfusionMatrixDisplay(
        confusion_matrix=
            gel_cm,

        display_labels=
            ratios_upper,
    )


    disp.plot(
        ax=ax,
        xticks_rotation=45,
        colorbar=False,
    )


    ax.set_title(
        f"{modality_name.upper()} "
        f"Z-SCORE LOSO-LDA\n"
        f"Gel-level accuracy = "
        f"{overall_gel_accuracy:.3f}"
    )


    plt.tight_layout()


    plt.savefig(
        os.path.join(
            save_path,
            f"{modality_name}_"
            f"zscore_gel_confusion_matrix.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )


    plt.close()


    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================

    importance_df = (
        pd.concat(
            importance_list,
            axis=1,
        )
    )


    importance_df[
        "mean_importance"
    ] = (
        importance_df.mean(
            axis=1
        )
    )


    fold_cols = [
        c
        for c in importance_df.columns
        if c.startswith(
            "fold_"
        )
    ]


    importance_df[
        "std_importance"
    ] = (
        importance_df[
            fold_cols
        ]
        .std(
            axis=1,
            ddof=1,
        )
    )


    importance_df = (
        importance_df
        .sort_values(
            "mean_importance",
            ascending=False,
        )
    )


    importance_df.to_csv(
        os.path.join(
            save_path,
            f"{modality_name}_"
            f"zscore_LDA_feature_importance.csv",
        )
    )


    # =====================================================
    # SUMMARY
    # =====================================================

    result = {

        "modality":
            modality_name,

        "overall_technical_accuracy":
            overall_technical_accuracy,

        "mean_fold_technical_accuracy":
            fold_df[
                "technical_accuracy"
            ].mean(),

        "std_fold_technical_accuracy":
            fold_df[
                "technical_accuracy"
            ].std(
                ddof=1
            ),

        "overall_gel_level_accuracy":
            overall_gel_accuracy,

        "mean_fold_gel_accuracy":
            fold_df[
                "gel_level_accuracy"
            ].mean(),

        "std_fold_gel_accuracy":
            fold_df[
                "gel_level_accuracy"
            ].std(
                ddof=1
            ),

        "n_gel_level_predictions":
            len(
                gel_vote_df
            ),
    }


    print("\n")
    print(
        f"{modality_name.upper()} SUMMARY"
    )

    for k, v in result.items():

        print(
            k,
            ":",
            v,
        )


    return result


# =========================================================
# LOAD / NORMALIZE ALL REPLICATES
# =========================================================

replicate_data = {}


for replicate_name, config in replicates.items():

    replicate_data[
        replicate_name
    ] = prepare_one_replicate(

        replicate_name=
            replicate_name,

        config=
            config,

        ratios_upper=
            ratios_upper,

        ratios=
            ratios,

        subfolders=
            subfolders,

        sample_per_group=
            sample_per_group,

        random_state=
            random_state,
    )


# =========================================================
# CHECK FEATURE COLUMNS
# =========================================================

rep_names = list(
    replicates.keys()
)


reference_rep = (
    rep_names[0]
)


for modality in [
    "shg_X",
    "spec_X",
    "fused_X",
]:

    reference_cols = list(
        replicate_data[
            reference_rep
        ][
            modality
        ].columns
    )

    for rep in rep_names[1:]:

        other_cols = list(
            replicate_data[
                rep
            ][
                modality
            ].columns
        )

        if (
            reference_cols
            != other_cols
        ):

            raise ValueError(
                f"Feature columns differ "
                f"between {reference_rep} "
                f"and {rep} for {modality}"
            )


# =========================================================
# COMBINE BIOLOGICAL REPLICATES
# =========================================================

all_shg_X = pd.concat(
    [
        replicate_data[
            r
        ][
            "shg_X"
        ]
        for r in replicates
    ],
    axis=0,
    ignore_index=True,
)


all_spec_X = pd.concat(
    [
        replicate_data[
            r
        ][
            "spec_X"
        ]
        for r in replicates
    ],
    axis=0,
    ignore_index=True,
)


all_fused_X = pd.concat(
    [
        replicate_data[
            r
        ][
            "fused_X"
        ]
        for r in replicates
    ],
    axis=0,
    ignore_index=True,
)


all_y = np.concatenate(
    [
        replicate_data[
            r
        ][
            "y"
        ]
        for r in replicates
    ]
)


all_groups = np.concatenate(
    [
        replicate_data[
            r
        ][
            "groups"
        ]
        for r in replicates
    ]
)


# =========================================================
# DATASET SUMMARY
# =========================================================

print("\n")
print("=" * 75)

print(
    "WITHIN-REPLICATE Z-SCORED DATASET"
)

print("=" * 75)


print(
    "SHG:",
    all_shg_X.shape
)

print(
    "MIRSI:",
    all_spec_X.shape
)

print(
    "Fused:",
    all_fused_X.shape
)

print(
    "Labels:",
    all_y.shape
)


print(
    "\nReplicate counts:"
)

print(
    pd.Series(
        all_groups
    ).value_counts()
)


# =========================================================
# SAVE AUDIT TABLE
# =========================================================

pd.DataFrame({

    "label":
        all_y,

    "biological_replicate":
        all_groups,

}).to_csv(

    os.path.join(
        save_path,
        "zscore_dataset_labels_and_replicates.csv",
    ),

    index=False,
)


# =========================================================
# RUN LDA
# =========================================================

summary_rows = []


for modality_name, X_mod in [

    (
        "shg_only",
        all_shg_X,
    ),

    (
        "spec_only",
        all_spec_X,
    ),

    (
        "fused",
        all_fused_X,
    ),
]:

    result = run_logo_lda(

        X=
            X_mod,

        y=
            all_y,

        groups=
            all_groups,

        modality_name=
            modality_name,

        save_path=
            save_path,

        ratios_upper=
            ratios_upper,

        use_pca=
            use_pca,

        pca_n_components=
            pca_n_components,
    )


    summary_rows.append(
        result
    )


# =========================================================
# FINAL SUMMARY
# =========================================================

summary_df = pd.DataFrame(
    summary_rows
)


summary_df.to_csv(
    os.path.join(
        save_path,
        "zscore_LOSO_LDA_summary.csv",
    ),
    index=False,
)


# =========================================================
# MANUSCRIPT TABLE
# =========================================================

table = (
    summary_df.copy()
)


table[
    "Modality"
] = table[
    "modality"
].map({

    "shg_only":
        "SHG",

    "spec_only":
        "MIRSI",

    "fused":
        "SHG + MIRSI",
})


table[
    "Technical Accuracy"
] = table[
    "overall_technical_accuracy"
].round(
    3
)


table[
    "Mean Technical Fold Accuracy"
] = table[
    "mean_fold_technical_accuracy"
].round(
    3
)


table[
    "Technical Fold SD"
] = table[
    "std_fold_technical_accuracy"
].round(
    3
)


table[
    "Gel-level Accuracy"
] = table[
    "overall_gel_level_accuracy"
].round(
    3
)


table[
    "Mean Gel Fold Accuracy"
] = table[
    "mean_fold_gel_accuracy"
].round(
    3
)


table[
    "Gel Fold SD"
] = table[
    "std_fold_gel_accuracy"
].round(
    3
)


table = table[
    [
        "Modality",
        "Technical Accuracy",
        "Mean Technical Fold Accuracy",
        "Technical Fold SD",
        "Gel-level Accuracy",
        "Mean Gel Fold Accuracy",
        "Gel Fold SD",
        "n_gel_level_predictions",
    ]
]


table.to_csv(
    os.path.join(
        save_path,
        "manuscript_table_zscore_LOSO_LDA.csv",
    ),
    index=False,
)


print("\n")
print("=" * 75)
print("DONE")
print("=" * 75)


print(
    table.to_string(
        index=False
    )
)


print(
    "\nSaved to:"
)

print(
    os.path.abspath(
        save_path
    )
)