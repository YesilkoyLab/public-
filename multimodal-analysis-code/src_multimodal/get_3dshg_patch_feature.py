import numpy as np
import pandas as pd
import tifffile as tf
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu, gaussian
from skimage.measure import label, regionprops
from skimage.feature import structure_tensor
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
import glob
import os
from pdb import set_trace as st

folder_path = "../data/03252026/processed/SHG"
folder_name = "03252026"
save_path = "../res/03252026"
tiff_files = sorted(glob.glob(os.path.join(folder_path, "*.tif")))

PATCH_SIZE = 32

# calibration
PIXEL_SIZE_UM = 0.33   # change if your XY pixel size is different
Z_STEP_UM = 1.0        # change if your z step is different

all_features = []

print(f"Found {len(tiff_files)} tif files")

for file in tiff_files:
    data = tf.imread(file)

    # only keep slice 3 to 17
    data = data[2:17]

    name = os.path.basename(file).split(".")[0]
    print(f"Loaded: {name}, shape={data.shape}")

    if data.shape[0] == 0:
        print(f"Skipped {name}: not enough slices.")
        continue

    Z, Y, X = data.shape

    for y0 in range(0, Y - PATCH_SIZE + 1, PATCH_SIZE):
        for x0 in range(0, X - PATCH_SIZE + 1, PATCH_SIZE):

            patch = data[:, y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE]
            data_smooth = gaussian(patch, sigma=1, preserve_range=True)

            # -----------------------------
            # Intensity features
            # -----------------------------
            mean_intensity = np.mean(data_smooth)
            std_intensity = np.std(data_smooth)
            max_intensity = np.max(data_smooth)
            intensity_sum = np.sum(data_smooth)

            # -----------------------------
            # Collagen density
            # -----------------------------
            try:
                threshold = threshold_otsu(data_smooth)
            except ValueError:
                threshold = np.mean(data_smooth)

            binary_mask = data_smooth > threshold
            collagen_density = np.sum(binary_mask) / binary_mask.size

            # -----------------------------
            # Coherence features
            # -----------------------------
            Axx, Axy, Axz, Ayy, Ayz, Azz = structure_tensor(data_smooth, sigma=1)

            tensor = np.array([
                [Axx, Axy, Axz],
                [Axy, Ayy, Ayz],
                [Axz, Ayz, Azz]
            ])

            tensor_reshaped = tensor.reshape(3, 3, -1).transpose(2, 0, 1)
            eigvals = np.linalg.eigvalsh(tensor_reshaped)
            eigvals = np.sort(eigvals, axis=1)[:, ::-1]
            eigvals = eigvals.reshape(*patch.shape, 3)

            lambda1 = eigvals[..., 0]
            lambda2 = eigvals[..., 1]
            lambda3 = eigvals[..., 2]

            coherence = (lambda1 - lambda2) / (
                lambda1 + lambda2 + lambda3 + 1e-8
            )

            mean_coherence = np.mean(coherence)

            # -----------------------------
            # Morphology features from skeleton
            # -----------------------------
            skeleton = skeletonize(binary_mask)

            label_img = label(skeleton)
            regions = regionprops(label_img)

            fiber_lengths = []
            fiber_straightness = []

            for region in regions:
                if region.area < 5:
                    continue

                try:
                    length = region.major_axis_length
                    width = region.minor_axis_length
                    straightness = width / (length + 1e-8)

                    fiber_lengths.append(length)
                    fiber_straightness.append(straightness)

                except ValueError:
                    continue

            mean_length = np.mean(fiber_lengths) if fiber_lengths else 0
            mean_straightness = np.mean(fiber_straightness) if fiber_straightness else 0

            # -----------------------------
            # Fiber diameter features
            # -----------------------------
            if np.sum(binary_mask) > 0 and np.sum(skeleton) > 0:

                dist_um = distance_transform_edt(
                    binary_mask,
                    sampling=(Z_STEP_UM, PIXEL_SIZE_UM, PIXEL_SIZE_UM)
                )

                radius_vals_um = dist_um[skeleton > 0]
                diameter_vals_um = 2.0 * radius_vals_um

                diameter_vals_um = diameter_vals_um[np.isfinite(diameter_vals_um)]
                diameter_vals_um = diameter_vals_um[diameter_vals_um > 0]

                if len(diameter_vals_um) > 0:
                    mean_fiber_diameter_um = np.mean(diameter_vals_um)
                    median_fiber_diameter_um = np.median(diameter_vals_um)
                    std_fiber_diameter_um = np.std(diameter_vals_um)
                    p90_fiber_diameter_um = np.percentile(diameter_vals_um, 90)
                else:
                    mean_fiber_diameter_um = 0
                    median_fiber_diameter_um = 0
                    std_fiber_diameter_um = 0
                    p90_fiber_diameter_um = 0

            else:
                mean_fiber_diameter_um = 0
                median_fiber_diameter_um = 0
                std_fiber_diameter_um = 0
                p90_fiber_diameter_um = 0

            # -----------------------------
            # Store feature vector
            # -----------------------------
            feature_vector = {
                "file_name": name,
                "x": x0,
                "y": y0,

                "mean_intensity": mean_intensity,
                "std_intensity": std_intensity,
                "max_intensity": max_intensity,
                "intensity_sum": intensity_sum,

                "collagen_density": collagen_density,
                "mean_coherence": mean_coherence,

                "mean_fiber_length": mean_length,
                "mean_fiber_straightness": mean_straightness,

                "mean_fiber_diameter_um": mean_fiber_diameter_um,
                "median_fiber_diameter_um": median_fiber_diameter_um,
                "std_fiber_diameter_um": std_fiber_diameter_um,
                "p90_fiber_diameter_um": p90_fiber_diameter_um,
            }

            all_features.append(feature_vector)


# -----------------------------
# Save as CSV
# -----------------------------
if len(all_features) == 0:
    print("No features extracted.")
    print("Check folder_path, tif files, or PATCH_SIZE.")
else:
    df = pd.DataFrame(all_features)

    cols = ["file_name", "x", "y"] + [
        col for col in df.columns if col not in ["file_name", "x", "y"]
    ]

    df = df[cols]

    os.makedirs(save_path, exist_ok=True)

    output_csv_path = os.path.join(
        save_path,
        f"{folder_name}_patch{PATCH_SIZE}_features_15slice.csv"
    )

    df.to_csv(output_csv_path, index=False)

    print(f"Features saved to: {output_csv_path}")