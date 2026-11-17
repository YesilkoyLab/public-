import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"

import numpy as np
import pandas as pd
import random
from glob import glob

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model


# -------------------------
# Read SHG features (bw or fw)
# -------------------------
def read_shg_data(shg_glcm_path, shg_morph_path, use_dir="bw"):
    glcm_df = pd.read_csv(shg_glcm_path)
    morph_df = pd.read_csv(shg_morph_path)
    merged_df = pd.merge(glcm_df, morph_df, on=["file_name", "x", "y"], how="inner")

    fw_df   = merged_df[merged_df["file_name"].str.endswith("-fw")].copy()
    bw_df   = merged_df[merged_df["file_name"].str.endswith("-bw")].copy()
    fw_df_1 = merged_df[merged_df["file_name"].str.endswith("-fw-1")].copy()
    bw_df_1 = merged_df[merged_df["file_name"].str.endswith("-bw-1")].copy()

    fw_df["base_name"]   = fw_df["file_name"].str.replace("-fw", "", regex=False)
    bw_df["base_name"]   = bw_df["file_name"].str.replace("-bw", "", regex=False)
    fw_df_1["base_name"] = fw_df_1["file_name"].str.replace("-fw-1", "", regex=False)
    bw_df_1["base_name"] = bw_df_1["file_name"].str.replace("-bw-1", "", regex=False)

    fw_df_combined = pd.concat([fw_df, fw_df_1], axis=0, ignore_index=True).add_suffix("_fw")
    bw_df_combined = pd.concat([bw_df, bw_df_1], axis=0, ignore_index=True).add_suffix("_bw")

    if use_dir.lower() == "bw":
        df = bw_df_combined
        df.rename(columns={"base_name_bw": "label"}, inplace=True)
        drop_cols = ["file_name_bw", "x_bw", "y_bw"]
    else:
        df = fw_df_combined
        df.rename(columns={"base_name_fw": "label"}, inplace=True)
        drop_cols = ["file_name_fw", "x_fw", "y_fw"]

    df.drop(columns=drop_cols, inplace=True, errors="ignore")

    feat_cols = [c for c in df.columns if c != "label"]
    df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=0).reset_index(drop=True)

    print("shg_df shape:", df.shape)
    return df


# -------------------------
# Read spectral data (npy files)
# -------------------------
def read_spec_data(spectral_root, ratios, subfolder, sample_per_group, seed=0):
    rng = random.Random(seed)
    spectral_data, labels = [], []

    for ratio in ratios:
        folder_path = os.path.join(spectral_root, ratio, subfolder)
        npy_files = sorted(glob(os.path.join(folder_path, "*.npy")))
        if len(npy_files) < sample_per_group:
            raise ValueError(f"{ratio}: only {len(npy_files)} npy files, need {sample_per_group}")

        sampled_files = rng.sample(npy_files, sample_per_group)
        for f in sampled_files:
            spectrum = np.load(f).astype(np.float32)
            spectral_data.append(spectrum)
            labels.append(str(ratio))

    df = pd.DataFrame(spectral_data)
    df["label"] = labels

    # rename 426 points 950..1800 step 2
    new_cols = list(range(950, 1802, 2))
    old_cols = df.columns[:len(new_cols)]
    df.rename(columns={o: n for o, n in zip(old_cols, new_cols)}, inplace=True)

    print("spec_df shape:", df.shape)
    return df


# -------------------------
# CAE (stable optimizer)
# -------------------------
def build_cae(input_dim: int, latent_dim: int = 8):
    inp = layers.Input(shape=(input_dim,), name="x")
    h = layers.Dense(128, activation="relu")(inp)
    h = layers.Dense(64, activation="relu")(h)
    z = layers.Dense(latent_dim, name="z")(h)

    h = layers.Dense(64, activation="relu")(z)
    h = layers.Dense(128, activation="relu")(h)
    out = layers.Dense(input_dim, name="x_hat")(h)

    cae = Model(inp, out)
    encoder = Model(inp, z)

    opt = tf.keras.optimizers.SGD(learning_rate=1e-2, momentum=0.9, nesterov=True)
    cae.compile(optimizer=opt, loss="mse")
    return cae, encoder


def make_chunk_groups(y, chunks_per_label=8):
    y = np.asarray(y)
    unique_labels = sorted(np.unique(y).tolist())
    groups = np.zeros(len(y), dtype=int)
    gid = 0
    for lab in unique_labels:
        idx = np.where(y == lab)[0]
        idx = np.sort(idx)
        chunk_size = int(np.ceil(len(idx) / chunks_per_label))
        for k in range(chunks_per_label):
            s = k * chunk_size
            e = min((k + 1) * chunk_size, len(idx))
            if s >= e:
                continue
            groups[idx[s:e]] = gid
            gid += 1
    return groups, unique_labels


def main():
    # --- config ---
    name = "11102025-zstack"
    path = f"../res/{name}/"
    shg_glcm_path = path + f"{name}_glcm_patch32_offset5.csv"
    shg_morph_path = path + f"{name}_patch32_features.csv"

    spectral_root = r"../../SPEC/res/Caf2_11102025_2nd"
    ratios = ["9010PSER", "8020PSER", "7030PSER", "6040PSER", "1000", "9010SER"]
    subfolder = "LMT_1"
    sample_per_group = 512
    seed = 0

    # --- load both modalities ---
    spec_df = read_spec_data(spectral_root, ratios, subfolder, sample_per_group, seed=seed)
    shg_df = read_shg_data(shg_glcm_path, shg_morph_path, use_dir="bw")

    # IMPORTANT: you must align rows between spec_df and shg_df.
    # If you don't have a shared patch id, the next best is to enforce SAME label order and take per-label blocks.
    # Here we assume shg_df is already ordered by label blocks consistent with ratios, and has >= sample_per_group rows per label.
    # We will build an aligned SHG subset by taking exactly 512 rows per label, in the same label order as spec_df.
    shg_df["label"] = shg_df["label"].astype(str).str.lower()
    spec_df["label"] = spec_df["label"].astype(str)
    spec_df["label_lower"] = spec_df["label"].str.lower()

    shg_blocks = []
    for lab in spec_df["label_lower"].unique():  # preserves encounter order (ratios order)
        block = shg_df[shg_df["label"] == lab].copy()
        if len(block) < sample_per_group:
            raise ValueError(f"SHG label {lab} has {len(block)} rows, need {sample_per_group}")
        shg_blocks.append(block.iloc[:sample_per_group])

    shg_aligned = pd.concat(shg_blocks, axis=0).reset_index(drop=True)
    # drop shg label
    shg_feat = shg_aligned.drop(columns=["label"]).reset_index(drop=True)

    # final fused df
    fused_df = pd.concat(
        [spec_df.drop(columns=["label_lower"]).reset_index(drop=True), shg_feat],
        axis=1
    )
    print("fused_df shape:", fused_df.shape)

    y = fused_df["label"].astype(str).values
    X = fused_df.drop(columns=["label"]).values.astype(np.float32)

    # optional: log1p on SHG columns only (recommended)
    # detect SHG columns as those not in wavenumber range (950..1800)
    wavenum_cols = set(range(950, 1802, 2))
    cols = fused_df.drop(columns=["label"]).columns
    is_shg = np.array([c not in wavenum_cols for c in cols], dtype=bool)
    X[:, is_shg] = np.log1p(np.clip(X[:, is_shg], a_min=0, a_max=None))

    groups, unique_labels = make_chunk_groups(y, chunks_per_label=8)
    cv = GroupKFold(n_splits=6)

    all_true, all_pred = [], []

    # Two-branch CAE: one for spectra, one for SHG, then concatenate latents
    spec_dim = len([c for c in fused_df.columns if isinstance(c, int)])  # wavenumber columns are ints
    shg_dim = X.shape[1] - spec_dim

    # Column indices
    feat_cols = fused_df.drop(columns=["label"]).columns
    spec_idx = np.array([isinstance(c, int) for c in feat_cols], dtype=bool)
    shg_idx = ~spec_idx

    latent_spec = 4
    latent_shg = 4

    for fold, (tr_idx, te_idx) in enumerate(cv.split(X, y, groups=groups), start=1):
        tf.keras.backend.clear_session()

        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # split modalities
        Xs_tr, Xh_tr = X_tr[:, spec_idx], X_tr[:, shg_idx]
        Xs_te, Xh_te = X_te[:, spec_idx], X_te[:, shg_idx]

        # scale each modality within fold
        ss = StandardScaler()
        sh = StandardScaler()
        Xs_tr_s = ss.fit_transform(Xs_tr)
        Xs_te_s = ss.transform(Xs_te)
        Xh_tr_s = sh.fit_transform(Xh_tr)
        Xh_te_s = sh.transform(Xh_te)

        # train CAEs
        cae_s, enc_s = build_cae(input_dim=Xs_tr_s.shape[1], latent_dim=latent_spec)
        cae_h, enc_h = build_cae(input_dim=Xh_tr_s.shape[1], latent_dim=latent_shg)

        cae_s.fit(Xs_tr_s, Xs_tr_s, epochs=120, batch_size=32, verbose=0,
                  callbacks=[tf.keras.callbacks.EarlyStopping(monitor="loss", patience=12, restore_best_weights=True)])
        cae_h.fit(Xh_tr_s, Xh_tr_s, epochs=120, batch_size=32, verbose=0,
                  callbacks=[tf.keras.callbacks.EarlyStopping(monitor="loss", patience=12, restore_best_weights=True)])

        # encode and fuse latents
        Zs_tr = enc_s.predict(Xs_tr_s, verbose=0)
        Zs_te = enc_s.predict(Xs_te_s, verbose=0)
        Zh_tr = enc_h.predict(Xh_tr_s, verbose=0)
        Zh_te = enc_h.predict(Xh_te_s, verbose=0)

        Z_tr = np.concatenate([Zs_tr, Zh_tr], axis=1)
        Z_te = np.concatenate([Zs_te, Zh_te], axis=1)

        # clf = LogisticRegression(max_iter=5000)
        # clf.fit(Z_tr, y_tr)
        # pred = clf.predict(Z_te)
        # Train separate classifiers Late fusion
        clf_spec = LogisticRegression(max_iter=5000)
        clf_shg  = LogisticRegression(max_iter=5000)

        clf_spec.fit(Zs_tr, y_tr)
        clf_shg.fit(Zh_tr, y_tr)

        # Probabilities
        ps = clf_spec.predict_proba(Zs_te)
        ph = clf_shg.predict_proba(Zh_te)

        # Align class order (important!)
        classes = clf_shg.classes_
        # ensure clf_spec uses same class order
        ps = ps[:, np.argsort(clf_spec.classes_)]
        ph = ph[:, np.argsort(clf_shg.classes_)]

        # weight (SHG dominates)
        w = 0.75
        pf = w * ph + (1 - w) * ps

        pred = classes[np.argmax(pf, axis=1)]

        all_true.extend(list(y_te))
        all_pred.extend(list(pred))
        print(f"Fold {fold} done")

    print("\n=== Fused (Spec + SHG): two-branch CAE-encoders + LogisticRegression (GroupKFold on chunks) ===")
    print("Accuracy:", accuracy_score(all_true, all_pred))
    print("\nConfusion matrix (labels order):", unique_labels)
    print(confusion_matrix(all_true, all_pred, labels=unique_labels))
    # print("\nReport:\n", classification_report(all_true, all_pred, labels=unique_labels))


if __name__ == "__main__":
    main()
