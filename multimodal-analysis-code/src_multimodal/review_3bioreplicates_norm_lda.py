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

name = "biological_replicate_LOSO_LDA_control_centered"

random_state = 42
random.seed(random_state)
np.random.seed(random_state)


# ---------------------------------------------------------
# Classes
# ---------------------------------------------------------

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

CONTROL_LABEL = "1000"


# ---------------------------------------------------------
# MIRSI technical folders / ROIs
# ---------------------------------------------------------

subfolders = [
    "LMT_1",
    "LMT_2",
    # "LMT_3",
]


# ---------------------------------------------------------
# Number of technical observations per formulation
# within each biological replicate
# ---------------------------------------------------------

sample_per_group = 128


# ---------------------------------------------------------
# PCA settings
# ---------------------------------------------------------

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
            "../res/03252026/03252026_glcm_patch32_offset5.csv",

        "shg_morph":
            "../res/03252026/03252026_patch32_features.csv",

        "spectral_root":
            r"../../spec_analysis/res/Caf2_03252026/2nd",
    },
}


# ---------------------------------------------------------
# Save directory
# ---------------------------------------------------------

save_path = "../res/biological_replicate_LOSO_LDA_control_centered"

os.makedirs(save_path, exist_ok=True)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def extract_label(file_name):
    """
    Example:

        8020PSER_1 -> 8020PSER
        9010SER_3  -> 9010SER
    """

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

    glcm_df = pd.read_csv(glcm_csv)
    morph_df = pd.read_csv(morph_csv)

    print(
        "GLCM shape:",
        glcm_df.shape,
    )

    print(
        "Morph shape:",
        morph_df.shape,
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
        merged["label"].isin(valid_labels)
    ].copy()

    print(
        "Merged SHG shape:",
        merged.shape,
    )

    print(
        "\nSHG observations per class:"
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

    """
    Sample MIRSI spectra approximately equally
    across the three LMT folders / ROIs.
    """

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

            spectrum = np.load(f)

            spectrum = (
                np.asarray(spectrum)
                .flatten()
            )

            if len(spectrum) != expected_length:

                raise ValueError(
                    "\nSpectrum length mismatch:\n"
                    f"{f}\n"
                    f"Expected: {expected_length}\n"
                    f"Found: {len(spectrum)}"
                )

            if np.any(
                ~np.isfinite(spectrum)
            ):
                raise ValueError(
                    f"Invalid spectrum:\n{f}"
                )

            spectral_rows.append({
                "file": f,
                "label": ratio.upper(),
                "subfolder": subfolder,
                "spectrum": spectrum,
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

    # 950–1800 cm^-1 at 2 cm^-1 intervals
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
        "\nMIRSI observations per class:"
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
# NUMERIC CLEANING
# =========================================================

def convert_numeric(df):

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
        [np.inf, -np.inf],
        np.nan,
    )

    return df


# =========================================================
# 100:0 CONTROL CENTERING
# =========================================================

def control_center(
    X,
    y,
    control_label="1000",
):

    """
    Normalize every feature relative to the mean
    100:0 control within the SAME biological replicate.

    X_centered = X - mean(X_control)

    No division is performed.

    Example:

        absolute feature = 4.8
        control mean     = 4.2
        centered feature = +0.6
    """

    X = convert_numeric(X)

    y = np.asarray(y)

    control_mask = (
        y == control_label
    )

    if control_mask.sum() == 0:

        raise ValueError(
            f"No control observations found "
            f"for label {control_label}"
        )

    control_mean = (
        X.loc[
            control_mask
        ]
        .mean(
            axis=0
        )
    )

    X_centered = (
        X - control_mean
    )

    return (
        X_centered,
        control_mean,
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
        f"PREPARING BIOLOGICAL REPLICATE: "
        f"{replicate_name}"
    )
    print("=" * 75)

    # -----------------------------------------------------
    # Load SHG
    # -----------------------------------------------------

    shg_df = load_shg_features(
        glcm_csv=
            config["shg_glcm"],

        morph_csv=
            config["shg_morph"],

        valid_labels=
            set(ratios_upper),
    )

    # -----------------------------------------------------
    # Load MIRSI
    # -----------------------------------------------------

    spec_df = load_spec_features(
        spectral_root=
            config["spectral_root"],

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

        if len(group_df) < sample_per_group:

            raise ValueError(
                "\nNot enough SHG samples.\n"
                f"Replicate: {replicate_name}\n"
                f"Class: {ratio}\n"
                f"Found: {len(group_df)}\n"
                f"Need: {sample_per_group}"
            )

        sampled = group_df.sample(
            n=sample_per_group,
            random_state=random_state,
        ).reset_index(
            drop=True
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

        if len(group_df) < sample_per_group:

            raise ValueError(
                "\nNot enough MIRSI samples.\n"
                f"Replicate: {replicate_name}\n"
                f"Class: {ratio}\n"
                f"Found: {len(group_df)}\n"
                f"Need: {sample_per_group}"
            )

        sampled = group_df.sample(
            n=sample_per_group,
            random_state=random_state,
        ).reset_index(
            drop=True
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
    # VERIFY CLASS ORDER
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
    # LABEL VECTOR
    # =====================================================

    y = spec_selected[
        "label"
    ].values


    # =====================================================
    # SHG FEATURE MATRIX
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
    # MIRSI FEATURE MATRIX
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
    # CONTROL-CENTER SHG
    # =====================================================

    (
        shg_X_centered,
        shg_control_mean,
    ) = control_center(
        X=
            shg_X_raw,

        y=
            y,

        control_label=
            CONTROL_LABEL,
    )


    # =====================================================
    # CONTROL-CENTER MIRSI
    # =====================================================

    (
        spec_X_centered,
        spec_control_mean,
    ) = control_center(
        X=
            spec_X_raw,

        y=
            y,

        control_label=
            CONTROL_LABEL,
    )


    # =====================================================
    # SAVE THE CONTROL REFERENCE VALUES
    # =====================================================

    control_dir = os.path.join(
        save_path,
        "control_reference_means",
    )

    os.makedirs(
        control_dir,
        exist_ok=True,
    )

    shg_control_mean.to_csv(
        os.path.join(
            control_dir,
            f"{replicate_name}_"
            f"SHG_1000_control_mean.csv",
        ),
        header=[
            "control_mean"
        ],
    )

    spec_control_mean.to_csv(
        os.path.join(
            control_dir,
            f"{replicate_name}_"
            f"MIRSI_1000_control_mean.csv",
        ),
        header=[
            "control_mean"
        ],
    )


    # =====================================================
    # SAVE CONTROL-CENTERED FEATURE TABLES
    # =====================================================

    centered_dir = os.path.join(
        save_path,
        "centered_features",
    )

    os.makedirs(
        centered_dir,
        exist_ok=True,
    )

    shg_save = (
        shg_X_centered.copy()
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
            centered_dir,
            f"{replicate_name}_"
            f"SHG_centered.csv",
        ),
        index=False,
    )


    spec_save = (
        spec_X_centered.copy()
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
            centered_dir,
            f"{replicate_name}_"
            f"MIRSI_centered.csv",
        ),
        index=False,
    )


    # =====================================================
    # FEATURE-LEVEL FUSION
    # =====================================================
    #
    # IMPORTANT:
    #
    # This is NOT pixel-level co-registration.
    #
    # SHG and MIRSI observations are paired only
    # within the same:
    #
    # biological replicate + formulation class
    #
    # =====================================================

    fused_X_centered = pd.concat(
        [
            spec_X_centered.reset_index(
                drop=True
            ),

            shg_X_centered.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )


    # =====================================================
    # BIOLOGICAL REPLICATE GROUP VARIABLE
    # =====================================================

    groups = np.array(
        [
            replicate_name
        ] * len(y)
    )


    # =====================================================
    # PRINT CHECK
    # =====================================================

    print(
        f"\n{replicate_name} feature shapes"
    )

    print(
        "SHG:",
        shg_X_centered.shape
    )

    print(
        "MIRSI:",
        spec_X_centered.shape
    )

    print(
        "Fused:",
        fused_X_centered.shape
    )


    # -----------------------------------------------------
    # Check that centered control approximately equals zero
    # -----------------------------------------------------

    control_mask = (
        y == CONTROL_LABEL
    )

    print(
        "\nMean centered SHG control "
        "(average absolute feature value):"
    )

    print(
        np.nanmean(
            np.abs(
                shg_X_centered.loc[
                    control_mask
                ].mean(
                    axis=0
                ).values
            )
        )
    )

    print(
        "\nMean centered MIRSI control "
        "(average absolute feature value):"
    )

    print(
        np.nanmean(
            np.abs(
                spec_X_centered.loc[
                    control_mask
                ].mean(
                    axis=0
                ).values
            )
        )
    )


    return {
        "shg_X":
            shg_X_centered,

        "spec_X":
            spec_X_centered,

        "fused_X":
            fused_X_centered,

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

    """
    Any missing values in both training and test data
    are filled using medians calculated ONLY from the
    training biological replicates.
    """

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

    # Entirely missing training columns
    bad_cols = X_train.columns[
        X_train.isna().any()
    ]

    if len(bad_cols) > 0:

        print(
            "\nDropping problematic columns:"
        )

        print(
            list(bad_cols)
        )

        X_train = (
            X_train.drop(
                columns=bad_cols
            )
        )

        X_test = (
            X_test.drop(
                columns=bad_cols
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

    predictions = list(
        predictions
    )

    return Counter(
        predictions
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
        index=feature_names,
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
        f"CONTROL-CENTERED LOSO-LDA: "
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
    # LOSO LOOP
    # =====================================================

    for fold, (
        train_idx,
        test_idx,
    ) in enumerate(
        logo.split(
            X,
            y,
            groups=groups,
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

        if len(test_reps) != 1:

            raise ValueError(
                "Expected exactly one "
                "held-out biological replicate."
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
            "Training replicates:",
            train_reps,
        )

        print(
            "Testing replicate:",
            test_rep,
        )


        # =================================================
        # TRAIN / TEST SPLIT
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
        # STANDARDIZATION
        #
        # IMPORTANT:
        #
        # Biological control-centering has already
        # happened separately for each replicate.
        #
        # StandardScaler here is MODEL PREPROCESSING.
        #
        # It is fitted ONLY on training biological
        # replicates.
        # =================================================

        scaler = StandardScaler()

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
        #
        # Fit ONLY on training replicates.
        # =================================================

        if use_pca:

            pca = PCA(
                n_components=
                    pca_n_components,
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
                X_train_proc.shape[1],
            )

            print(
                "Explained variance:",
                round(
                    pca.explained_variance_ratio_
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
        # TECHNICAL-OBSERVATION ACCURACY
        # =================================================

        fold_technical_acc = (
            accuracy_score(
                y_test,
                y_pred,
            )
        )

        print(
            "Technical-observation accuracy:",
            round(
                fold_technical_acc,
                4,
            ),
        )


        # =================================================
        # SAVE INDIVIDUAL PREDICTIONS
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
        # GEL / FORMULATION LEVEL MAJORITY VOTE
        # =================================================
        #
        # Each class within each held-out biological
        # replicate receives ONE final prediction.
        #
        # 9 classes x 3 biological replicates
        # = 27 gel-level predictions.
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

            if len(class_df) == 0:
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
                    len(class_df),
            })


        fold_gel_accuracy = (
            np.mean(
                fold_gel_correct
            )
        )

        print(
            "Gel/formulation-level accuracy:",
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
    # SAVE FOLD RESULTS
    # =====================================================

    fold_df = pd.DataFrame(
        fold_rows
    )

    fold_df.to_csv(
        os.path.join(
            save_path,
            f"{modality_name}_"
            f"control_centered_"
            f"LOSO_fold_results.csv",
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
            f"control_centered_"
            f"technical_predictions.csv",
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
            f"control_centered_"
            f"gel_level_predictions.csv",
        ),
        index=False,
    )


    # =====================================================
    # TECHNICAL-LEVEL CONFUSION MATRIX
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

    technical_cm_df = (
        pd.DataFrame(
            technical_cm,
            index=
                ratios_upper,
            columns=
                ratios_upper,
        )
    )

    technical_cm_df.to_csv(
        os.path.join(
            save_path,
            f"{modality_name}_"
            f"control_centered_"
            f"technical_confusion_matrix.csv",
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
        f"CONTROL-CENTERED LOSO-LDA\n"
        f"Technical-observation accuracy = "
        f"{overall_technical_accuracy:.3f}"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            save_path,
            f"{modality_name}_"
            f"control_centered_"
            f"technical_confusion_matrix.png",
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

    gel_cm = confusion_matrix(
        gel_true,
        gel_pred,
        labels=
            ratios_upper,
    )

    gel_cm_df = pd.DataFrame(
        gel_cm,
        index=
            ratios_upper,
        columns=
            ratios_upper,
    )

    gel_cm_df.to_csv(
        os.path.join(
            save_path,
            f"{modality_name}_"
            f"control_centered_"
            f"gel_level_confusion_matrix.csv",
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
        f"CONTROL-CENTERED LOSO-LDA\n"
        f"Gel-level accuracy = "
        f"{overall_gel_accuracy:.3f}"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            save_path,
            f"{modality_name}_"
            f"control_centered_"
            f"gel_level_confusion_matrix.png",
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
    ] = importance_df.mean(
        axis=1
    )

    fold_importance_cols = [
        c
        for c in importance_df.columns
        if c.startswith(
            "fold_"
        )
    ]

    importance_df[
        "std_importance"
    ] = importance_df[
        fold_importance_cols
    ].std(
        axis=1,
        ddof=1,
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
            f"control_centered_"
            f"LDA_feature_importance.csv",
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
# LOAD AND CONTROL-CENTER ALL BIOLOGICAL REPLICATES
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
# CHECK THAT FEATURE COLUMNS MATCH
# =========================================================

rep_names = list(
    replicates.keys()
)

reference_rep = rep_names[0]


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

        if reference_cols != other_cols:

            raise ValueError(
                f"Feature columns do not match "
                f"between {reference_rep} and {rep} "
                f"for {modality}"
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
# DATASET CHECK
# =========================================================

print("\n")
print("=" * 75)
print(
    "CONTROL-CENTERED COMBINED DATASET"
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
    "\nBiological replicate counts:"
)

print(
    pd.Series(
        all_groups
    ).value_counts()
)


# =========================================================
# SAVE LABEL / REPLICATE AUDIT TABLE
# =========================================================

audit_df = pd.DataFrame({

    "label":
        all_y,

    "biological_replicate":
        all_groups,
})

audit_df.to_csv(
    os.path.join(
        save_path,
        "control_centered_"
        "dataset_labels_and_replicates.csv",
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
        "control_centered_"
        "LOSO_LDA_summary.csv",
    ),
    index=False,
)


# =========================================================
# MANUSCRIPT-READY TABLE
# =========================================================

manuscript_table = (
    summary_df.copy()
)


manuscript_table[
    "Modality"
] = manuscript_table[
    "modality"
].map({

    "shg_only":
        "SHG",

    "spec_only":
        "MIRSI",

    "fused":
        "SHG + MIRSI",
})


manuscript_table[
    "Technical Accuracy"
] = manuscript_table[
    "overall_technical_accuracy"
].round(
    3
)


manuscript_table[
    "Mean Technical Fold Accuracy"
] = manuscript_table[
    "mean_fold_technical_accuracy"
].round(
    3
)


manuscript_table[
    "Technical Fold SD"
] = manuscript_table[
    "std_fold_technical_accuracy"
].round(
    3
)


manuscript_table[
    "Gel-level Accuracy"
] = manuscript_table[
    "overall_gel_level_accuracy"
].round(
    3
)


manuscript_table[
    "Mean Gel Fold Accuracy"
] = manuscript_table[
    "mean_fold_gel_accuracy"
].round(
    3
)


manuscript_table[
    "Gel Fold SD"
] = manuscript_table[
    "std_fold_gel_accuracy"
].round(
    3
)


manuscript_table = manuscript_table[
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


manuscript_table.to_csv(
    os.path.join(
        save_path,
        "manuscript_table_"
        "control_centered_LOSO_LDA.csv",
    ),
    index=False,
)


# =========================================================
# PRINT FINAL RESULTS
# =========================================================

print("\n")
print("=" * 75)
print("DONE")
print("=" * 75)

print(
    manuscript_table.to_string(
        index=False
    )
)

print(
    "\nResults saved to:"
)

print(
    os.path.abspath(
        save_path
    )
)