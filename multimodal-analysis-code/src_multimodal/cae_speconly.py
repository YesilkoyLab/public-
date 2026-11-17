import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
import numpy as np
import pandas as pd
import random
from glob import glob
from tensorflow.keras.optimizers.legacy import Adam 
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import tensorflow as tf
from tensorflow.keras import layers, Model


# -------------------------
# Read ONLY spectral data
# -------------------------
def read_spec_data(spectral_root, ratios, subfolder, sample_per_group, seed=0):
    rng = random.Random(seed)

    spectral_data = []
    labels = []

    for ratio in ratios:
        folder_path = os.path.join(spectral_root, ratio, subfolder)
        npy_files = sorted(glob(os.path.join(folder_path, "*.npy")))
        if len(npy_files) < sample_per_group:
            raise ValueError(f"{ratio}: only {len(npy_files)} npy files, need {sample_per_group}")

        sampled_files = rng.sample(npy_files, sample_per_group)
        for f in sampled_files:
            spectrum = np.load(f).astype(np.float32)  # shape (426,) expected
            spectral_data.append(spectrum)
            labels.append(str(ratio))

    spectral_df = pd.DataFrame(spectral_data)
    spectral_df["label"] = labels

    # Rename spectral columns 950..1800 step 2 (426 points)
    new_column_names = list(range(950, 1802, 2))  # 950..1800 inclusive
    if spectral_df.shape[1] - 1 != len(new_column_names):
        # -1 because label column
        print("WARNING: spectral length != 426. Got:", spectral_df.shape[1] - 1)
    old_cols = spectral_df.columns[:len(new_column_names)]
    rename_dict = {old: new for old, new in zip(old_cols, new_column_names)}
    spectral_df.rename(columns=rename_dict, inplace=True)

    print("spectral_df.shape:", spectral_df.shape)
    return spectral_df


# -------------------------
# CAE (fully-connected)
# -------------------------
def build_cae(input_dim: int, latent_dim: int = 8):
    inp = layers.Input(shape=(input_dim,), name="x")

    # Encoder
    h = layers.Dense(128, activation="relu")(inp)
    h = layers.Dense(64, activation="relu")(h)
    z = layers.Dense(latent_dim, name="z")(h)

    # Decoder
    h = layers.Dense(64, activation="relu")(z)
    h = layers.Dense(128, activation="relu")(h)
    out = layers.Dense(input_dim, name="x_hat")(h)

    cae = Model(inp, out, name="cae")
    encoder = Model(inp, z, name="encoder")

    # SGD is very stable in Keras 3 CV loops on Windows
    opt = tf.keras.optimizers.SGD(learning_rate=1e-2, momentum=0.9, nesterov=True)
    cae.compile(optimizer=opt, loss="mse")
    return cae, encoder



def main():
    # --- Paths / config ---
    spectral_root = r"../../SPEC/res/Caf2_03272026/2nd"
    ratios = ["1000", "9010PSER", "8020PSER", "7030PSER", "6040PSER", "9010SER", "8020SER", "7030SER", "6040SER"]
    subfolder = "LMT_1"
    sample_per_group = 128

    # --- Load spectral only ---
    df = read_spec_data(spectral_root, ratios, subfolder, sample_per_group, seed=0)

    y = df["label"].astype(str).values
    X = df.drop(columns=["label"]).values.astype(np.float32)

    unique_labels = sorted(np.unique(y).tolist())
    print("X shape:", X.shape, "labels:", unique_labels)

    # -------------------------
    # Chunk-based GroupKFold CV
    # (so each fold contains all labels; avoids "leave-one-class-out" problem)
    # -------------------------
    chunks_per_label = 8  # 512 -> 8 chunks of 64 each
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

    cv = GroupKFold(n_splits=6)

    # --- CV loop ---
    all_true, all_pred = [], []
    input_dim = X.shape[1]
    latent_dim = 4  # try 4 or 8

    for fold, (tr_idx, te_idx) in enumerate(cv.split(X, y, groups=groups), start=1):
        tf.keras.backend.clear_session()

        
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # scale within fold
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # train CAE on training only
        cae, encoder = build_cae(input_dim=input_dim, latent_dim=latent_dim)
        cae.fit(
            X_tr_s, X_tr_s,
            epochs=120,
            batch_size=32,
            verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="loss", patience=12, restore_best_weights=True)]
        )

        # encode and classify subROIs directly
        Z_tr = encoder.predict(X_tr_s, verbose=0)
        Z_te = encoder.predict(X_te_s, verbose=0)
    
        clf = LogisticRegression(max_iter=5000)
        clf.fit(Z_tr, y_tr)
        pred = clf.predict(Z_te)

        all_true.extend(list(y_te))
        all_pred.extend(list(pred))
        print(f"Fold {fold} done")

    print("\n=== Spectral-only: CAE-encoder + LogisticRegression (GroupKFold on chunks) ===")
    print("Accuracy:", accuracy_score(all_true, all_pred))
    print("\nConfusion matrix (labels order):", unique_labels)
    print(confusion_matrix(all_true, all_pred, labels=unique_labels))
    print("\nReport:\n", classification_report(all_true, all_pred, labels=unique_labels))


if __name__ == "__main__":
    main()
