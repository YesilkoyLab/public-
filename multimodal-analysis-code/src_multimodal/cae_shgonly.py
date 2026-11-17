import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import tensorflow as tf
from tensorflow.keras import layers, Model


# -------------------------
# Read SHG-only features
# (your CSVs -> merged -> pick bw (or fw) -> label)
# -------------------------
def read_shg_data(shg_glcm_path, shg_morph_path, use_dir="bw"):
    glcm_df = pd.read_csv(shg_glcm_path)
    morph_df = pd.read_csv(shg_morph_path)

    merged_df = pd.merge(glcm_df, morph_df, on=["file_name", "x", "y"], how="inner")

    # Split fw/bw and fw-1/bw-1
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
        shg_df = bw_df_combined
        shg_df.rename(columns={"base_name_bw": "label"}, inplace=True)
        drop_cols = ["file_name_bw", "x_bw", "y_bw"]
    else:
        shg_df = fw_df_combined
        shg_df.rename(columns={"base_name_fw": "label"}, inplace=True)
        drop_cols = ["file_name_fw", "x_fw", "y_fw"]

    shg_df.drop(columns=drop_cols, inplace=True, errors="ignore")

    # Keep only numeric columns + label
    feat_cols = [c for c in shg_df.columns if c != "label"]
    shg_df[feat_cols] = shg_df[feat_cols].apply(pd.to_numeric, errors="coerce")
    shg_df = shg_df.dropna(axis=0).reset_index(drop=True)

    print("shg_df shape:", shg_df.shape)
    return shg_df


# -------------------------
# CAE (fully-connected)
# -------------------------
def build_cae(input_dim: int, latent_dim: int = 8):
    inp = layers.Input(shape=(input_dim,), name="x")

    h = layers.Dense(128, activation="relu")(inp)
    h = layers.Dense(64, activation="relu")(h)
    z = layers.Dense(latent_dim, name="z")(h)

    h = layers.Dense(64, activation="relu")(z)
    h = layers.Dense(128, activation="relu")(h)
    out = layers.Dense(input_dim, name="x_hat")(h)

    cae = Model(inp, out, name="cae")
    encoder = Model(inp, z, name="encoder")

    # stable in Keras 3 loops
    opt = tf.keras.optimizers.SGD(learning_rate=1e-2, momentum=0.9, nesterov=True)
    cae.compile(optimizer=opt, loss="mse")
    return cae, encoder


def make_chunk_groups(y, chunks_per_label=8):
    """Create chunk groups within each label so every fold contains all labels."""
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
    # --- Paths / config ---
    name = "02092026"
    path = f"../res/{name}/"
    shg_glcm_path = path + f"{name}_glcm_patch32_offset5.csv"
    shg_morph_path = path + f"{name}_patch32_features.csv"

    # --- Load SHG only ---
    df = read_shg_data(shg_glcm_path, shg_morph_path, use_dir="bw")

    y = df["label"].astype(str).values
    X = df.drop(columns=["label"]).values.astype(np.float32)

    # Optional: log1p on heavy-tailed metrics (often helps SHG)
    # Apply only to positive-valued features to avoid NaNs.
    X = np.log1p(np.clip(X, a_min=0, a_max=None)).astype(np.float32)

    groups, unique_labels = make_chunk_groups(y, chunks_per_label=8)
    print("X shape:", X.shape, "labels:", unique_labels)

    cv = GroupKFold(n_splits=6)

    all_true, all_pred = [], []
    input_dim = X.shape[1]
    latent_dim = 4  # try 4 or 8

    for fold, (tr_idx, te_idx) in enumerate(cv.split(X, y, groups=groups), start=1):
        tf.keras.backend.clear_session()

        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        cae, encoder = build_cae(input_dim=input_dim, latent_dim=latent_dim)
        cae.fit(
            X_tr_s, X_tr_s,
            epochs=120,
            batch_size=32,
            verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="loss", patience=12, restore_best_weights=True)]
        )

        Z_tr = encoder.predict(X_tr_s, verbose=0)
        Z_te = encoder.predict(X_te_s, verbose=0)

        clf = LogisticRegression(max_iter=5000)
        clf.fit(Z_tr, y_tr)
        pred = clf.predict(Z_te)

        all_true.extend(list(y_te))
        all_pred.extend(list(pred))
        print(f"Fold {fold} done")

    print("\n=== SHG-only: CAE-encoder + LogisticRegression (GroupKFold on chunks) ===")
    print("Accuracy:", accuracy_score(all_true, all_pred))
    print("\nConfusion matrix (labels order):", unique_labels)
    print(confusion_matrix(all_true, all_pred, labels=unique_labels))
    print("\nReport:\n", classification_report(all_true, all_pred, labels=unique_labels))


if __name__ == "__main__":
    main()
