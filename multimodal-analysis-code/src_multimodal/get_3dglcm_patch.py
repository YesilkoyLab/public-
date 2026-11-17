import os
import time
from glob import glob

import numpy as np
import pandas as pd
import tifffile as tf


# -----------------------------
# Fast, vectorized 3D GLCM
# -----------------------------
def glcm_3d_fast(arr: np.ndarray, delta: tuple[int, int, int], d: int) -> np.ndarray:
    """
    Vectorized 3D GLCM for integer 3D array.
    arr shape: (Z, Y, X) or any 3D; we treat axes as (a0, a1, a2).
    delta: (da0, da1, da2) direction
    d: offset distance multiplier
    Returns: levels x levels matrix normalized by levels^2 (to match your original behavior).
    """
    if arr.ndim != 3:
        raise ValueError("Input should be 3 dimensional")
    if "int" not in str(arr.dtype):
        raise ValueError("Input should be of dtype Int")

    da0, da1, da2 = (delta[0] * d, delta[1] * d, delta[2] * d)

    # determine overlapping valid region between A and shifted B
    a0_start = max(0, -da0)
    a0_end = arr.shape[0] - max(0, da0)
    a1_start = max(0, -da1)
    a1_end = arr.shape[1] - max(0, da1)
    a2_start = max(0, -da2)
    a2_end = arr.shape[2] - max(0, da2)

    if a0_end <= a0_start or a1_end <= a1_start or a2_end <= a2_start:
        # no overlap -> empty matrix
        levels = int(arr.max()) + 1
        return np.zeros((levels, levels), dtype=np.float64)

    A = arr[a0_start:a0_end, a1_start:a1_end, a2_start:a2_end]
    B = arr[a0_start + da0 : a0_end + da0, a1_start + da1 : a1_end + da1, a2_start + da2 : a2_end + da2]

    levels = int(arr.max()) + 1

    # flatten pairs into 1D index and bincount
    # idx = A * levels + B
    idx = (A.reshape(-1).astype(np.int64) * levels + B.reshape(-1).astype(np.int64))
    counts = np.bincount(idx, minlength=levels * levels).reshape(levels, levels).astype(np.float64)

    # keep your original normalization (note: not probability unless you prefer counts/sum(counts))
    return counts / (levels**2)


# -----------------------------
# GLCM properties (safe)
# -----------------------------
def graycoprops_3d(P: np.ndarray) -> dict:
    """
    Computes energy, homogeneity, contrast, entropy, correlation.
    Safely handles empty/zero-sum matrices.
    """
    s = float(np.sum(P))
    if s <= 0:
        return {"energy": 0.0, "homogeneity": 0.0, "contrast": 0.0, "entropy": 0.0, "correlation": 0.0}

    P = P / s
    energy = float(np.sum(P**2))

    idx = np.where(P > 0)
    i = idx[0].astype(np.float64) + 1.0
    j = idx[1].astype(np.float64) + 1.0
    p = P[idx].astype(np.float64)

    homogeneity = float(np.sum(p / (1.0 + (i - j) ** 2)))
    contrast = float(np.sum(p * (i - j) ** 2))
    entropy = float(np.sum(-p * np.log(p)))

    mu = float(np.sum(i * p))
    sigma_square = float(np.sum(p * (i - mu) ** 2))
    correlation = float(np.sum((p * (i - mu) * (j - mu)) / (sigma_square + 1e-8)))

    return {
        "energy": energy,
        "homogeneity": homogeneity,
        "contrast": contrast,
        "entropy": entropy,
        "correlation": correlation,
    }


# -----------------------------
# Feature extraction per TIFF
# -----------------------------
def extract_glcm_features(
    file_path: str,
    offset: int = 3,
    patch_size: int = 64,
    directions: list | None = None,
    progress_every: int = 25,
    z_range: tuple = (2, 17),   # 👈 新增
) -> list[dict]:

    img = tf.imread(file_path).astype(np.uint8)

    # apply slice range
    img = img[z_range[0]:z_range[1]]

    if img.shape[0] == 0:
        print(f"Skipped {file_path}: not enough slices.")
        return []
    if img.ndim != 3:
        raise ValueError(f"Expected 3D TIFF, got shape {img.shape} from {file_path}")

    Z, Y, X = img.shape
    name = os.path.basename(file_path).rsplit(".", 1)[0]

    if directions is None:
        directions = [
            ("a0", (1, 0, 0)),  # axis-0 direction (Z if your data is (Z,Y,X))
            ("a1", (0, 1, 0)),  # axis-1 direction
            ("a2", (0, 0, 1)),  # axis-2 direction
            ("d111", (1, 1, 1)),
            ("a0a1", (1, 1, 0)),
            ("a0a2", (1, 0, 1)),
            ("a1a2", (0, 1, 1)),
        ]

    nY = (Y - patch_size + 1) // patch_size
    nX = (X - patch_size + 1) // patch_size
    total = nY * nX

    patch_features: list[dict] = []
    t0 = time.time()
    patch_id = 0

    for y0 in range(0, Y - patch_size + 1, patch_size):
        for x0 in range(0, X - patch_size + 1, patch_size):
            patch_id += 1
            if progress_every and (patch_id % progress_every == 0):
                print(
                    f"    {name}: patch {patch_id}/{total} (y={y0}, x={x0}) "
                    f"elapsed {time.time() - t0:.1f}s"
                )

            patch = img[:, y0 : y0 + patch_size, x0 : x0 + patch_size]
            feats: dict = {"file_name": name, "x": int(x0), "y": int(y0)}

            # per-direction GLCM props
            for label, delta in directions:
                glcm_mat = glcm_3d_fast(patch, delta, offset)

                # match your original behavior: zero-out level 0
                if glcm_mat.shape[0] > 0:
                    glcm_mat[0, :] = 0
                    glcm_mat[:, 0] = 0

                props = graycoprops_3d(glcm_mat)
                for k, v in props.items():
                    feats[f"{label}_{k}"] = v

            # safe mean intensity on non-zero voxels
            nz = patch[patch > 0]
            feats["mean_intensity"] = float(np.mean(nz)) if nz.size else 0.0

            patch_features.append(feats)

    print(f"    {name}: done ({total} patches), time {time.time() - t0:.1f}s")
    return patch_features


def main():
    # ---- paths ----
    folder_path = r"../data/04202026/processed"
    folder_name = "04202026"
    save_path = r"../res-15/04202026"
    os.makedirs(save_path, exist_ok=True)

    # ---- params ----
    offset = 5
    patch_size = 32
    progress_every = 32  # print every N patches (set 0 to disable)

    tif_files = sorted(glob(os.path.join(folder_path, "*.tif")))
    print("Found tif files:", len(tif_files))
    if len(tif_files) == 0:
        print("No .tif files found. Check folder_path.")
        return

    save_csv_path = os.path.join(save_path, f"{folder_name}_glcm_patch{patch_size}_offset{offset}_15slice.csv")
    first = True

    for i, f in enumerate(tif_files):
        print(f"{i+1}/{len(tif_files)}: {os.path.basename(f)}")

        patch_feats = extract_glcm_features(
            f,
            offset=offset,
            patch_size=patch_size,
            progress_every=progress_every,
            z_range=(2,17),   # 👈 控制这里就行
        )

        df = pd.DataFrame(patch_feats)
        cols = ["file_name", "x", "y"] + [c for c in df.columns if c not in ["file_name", "x", "y"]]
        df = df[cols]

        df.to_csv(save_csv_path, mode="w" if first else "a", header=first, index=False)
        first = False

        # help memory a bit between files
        del df, patch_feats

    print(f"Saved features to {save_csv_path}")


if __name__ == "__main__":
    main()
