import numpy as np
import pandas as pd
import tifffile as tf
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu, gaussian
from skimage.measure import label, regionprops
from skimage.feature import structure_tensor
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt, convolve
import glob
import os

folder_path = "../data/04202026/processed"
folder_name = "04202026"
save_path = "../res-15/04202026"
tiff_files = sorted(glob.glob(os.path.join(folder_path, "*.tif")))

PATCH_SIZE = 32

# -----------------------------
# Voxel calibration
# -----------------------------
PIXEL_SIZE_UM = 0.33   # XY pixel size in um
Z_STEP_UM = 1.0        # Z spacing in um

EPS = 1e-8
all_features = []


# =========================================================
# Helper functions
# =========================================================
def compute_lacunarity_proxy(binary_mask_2d):
    h, w = binary_mask_2d.shape
    box_sizes = [2, 4, 8]
    vals = []

    for box in box_sizes:
        if h < box or w < box:
            continue

        masses = []
        for y in range(0, h - box + 1, box):
            for x in range(0, w - box + 1, box):
                sub = binary_mask_2d[y:y+box, x:x+box]
                masses.append(np.sum(sub))

        masses = np.array(masses, dtype=np.float64)
        if len(masses) == 0:
            continue

        mu = np.mean(masses)
        var = np.var(masses)
        if mu > 0:
            lac = var / (mu**2 + EPS) + 1.0
            vals.append(lac)

    return float(np.mean(vals)) if len(vals) > 0 else 0.0


def compute_skeleton_graph_features(skeleton):
    skeleton_uint = skeleton.astype(np.uint8)

    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    kernel[1, 1, 1] = 0

    neighbor_count = convolve(skeleton_uint, kernel, mode="constant", cval=0)

    endpoints = np.logical_and(skeleton, neighbor_count == 1)
    branchpoints = np.logical_and(skeleton, neighbor_count >= 3)

    endpoint_count = int(np.sum(endpoints))
    branchpoint_count = int(np.sum(branchpoints))
    skeleton_voxels = int(np.sum(skeleton))

    return endpoint_count, branchpoint_count, skeleton_voxels


def compute_spacing_features(binary_mask, sampling=(1.0, 1.0, 1.0)):
    inv_mask = ~binary_mask
    if np.sum(inv_mask) == 0:
        return 0.0, 0.0

    dist = distance_transform_edt(inv_mask, sampling=sampling)
    dist_vals = dist[inv_mask]

    mean_spacing = float(2.0 * np.mean(dist_vals)) if len(dist_vals) > 0 else 0.0
    median_spacing = float(2.0 * np.median(dist_vals)) if len(dist_vals) > 0 else 0.0

    return mean_spacing, median_spacing


def compute_fiber_diameter_features(binary_mask, skeleton, sampling=(1.0, 1.0, 1.0)):
    """
    Estimate local fiber diameter using distance transform.

    distance_transform_edt(binary_mask) gives the distance from each collagen voxel
    to the nearest background voxel. Along the skeleton, this distance approximates
    local fiber radius. Diameter = 2 * radius.

    Returns diameter features in um.
    """
    if np.sum(binary_mask) == 0 or np.sum(skeleton) == 0:
        return 0.0, 0.0, 0.0, 0.0

    dist_um = distance_transform_edt(binary_mask, sampling=sampling)

    radius_vals_um = dist_um[skeleton > 0]
    diameter_vals_um = 2.0 * radius_vals_um

    diameter_vals_um = diameter_vals_um[np.isfinite(diameter_vals_um)]
    diameter_vals_um = diameter_vals_um[diameter_vals_um > 0]

    if len(diameter_vals_um) == 0:
        return 0.0, 0.0, 0.0, 0.0

    mean_diameter = float(np.mean(diameter_vals_um))
    median_diameter = float(np.median(diameter_vals_um))
    std_diameter = float(np.std(diameter_vals_um))
    p90_diameter = float(np.percentile(diameter_vals_um, 90))

    return mean_diameter, median_diameter, std_diameter, p90_diameter


def compute_orientation_features_from_projection(img2d):
    if np.all(img2d == img2d.flat[0]):
        return 0.0, 0.0

    Axx, Axy, Ayy = structure_tensor(img2d, sigma=1)
    gx2 = np.mean(Axx)
    gy2 = np.mean(Ayy)
    gxy = np.mean(Axy)

    theta = 0.5 * np.arctan2(2 * gxy, gx2 - gy2)
    dominant_orientation_deg = float(np.degrees(theta) % 180)

    local_theta = 0.5 * np.arctan2(2 * Axy, Axx - Ayy + EPS)
    local_deg = np.degrees(local_theta) % 180

    ang = np.deg2rad(local_deg * 2.0)
    R = np.sqrt(np.mean(np.cos(ang))**2 + np.mean(np.sin(ang))**2)
    orientation_dispersion_deg = float(
        np.degrees(np.sqrt(max(0, 2 * (1 - R)) / 2.0))
    )

    return dominant_orientation_deg, orientation_dispersion_deg


def compute_fft_features(img2d):
    img2d = img2d.astype(np.float64)
    if np.allclose(img2d, img2d.flat[0]):
        return 0.0, 0.0, 0.0, 0.0

    img2d = img2d - np.mean(img2d)
    F = np.fft.fftshift(np.fft.fft2(img2d))
    P = np.abs(F)**2

    h, w = P.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.indices((h, w))
    ry = yy - cy
    rx = xx - cx
    rr = np.sqrt(rx**2 + ry**2)
    theta = (np.degrees(np.arctan2(ry, rx)) + 180) % 180

    valid = rr > 2
    P_valid = P[valid]

    fft_mean_power = float(np.mean(P_valid)) if np.any(valid) else 0.0

    rmax = np.max(rr)
    low_mask = np.logical_and(valid, rr <= 0.20 * rmax)
    high_mask = np.logical_and(valid, rr >= 0.50 * rmax)

    low_power = np.mean(P[low_mask]) if np.any(low_mask) else 0.0
    high_power = np.mean(P[high_mask]) if np.any(high_mask) else 0.0
    fft_radial_low_high_ratio = float(low_power / (high_power + EPS))

    angle_bins = np.linspace(0, 180, 19)
    angular_power = []

    for i in range(len(angle_bins) - 1):
        amask = np.logical_and(theta >= angle_bins[i], theta < angle_bins[i + 1])
        amask = np.logical_and(amask, valid)
        angular_power.append(np.mean(P[amask]) if np.any(amask) else 0.0)

    angular_power = np.array(angular_power, dtype=np.float64)

    if np.mean(angular_power) > 0:
        fft_anisotropy = float(
            (np.max(angular_power) - np.min(angular_power)) /
            (np.mean(angular_power) + EPS)
        )
    else:
        fft_anisotropy = 0.0

    peak_bin = int(np.argmax(angular_power))
    fft_peak_orientation_deg = float(
        (angle_bins[peak_bin] + angle_bins[peak_bin + 1]) / 2.0
    )

    return fft_mean_power, fft_radial_low_high_ratio, fft_anisotropy, fft_peak_orientation_deg


def compute_component_features(binary_mask):
    lbl = label(binary_mask)
    props = regionprops(lbl)

    if len(props) == 0:
        return 0, 0.0

    sizes = np.array([p.area for p in props], dtype=np.float64)
    total = np.sum(sizes)
    largest_fraction = float(np.max(sizes) / (total + EPS))

    return int(len(props)), largest_fraction


# =========================================================
# Main loop
# =========================================================
for file in tiff_files:
    data = tf.imread(file)

    # select slice 3–17
    data = data[2:17]

    name = os.path.basename(file).split(".")[0]
    print(f"Loaded: {name}")

    if data.shape[0] < 5:
        print(f"Warning: {name} has too few slices after selection")
        continue

    Z, Y, X = data.shape

    for y0 in range(0, Y - PATCH_SIZE + 1, PATCH_SIZE):
        for x0 in range(0, X - PATCH_SIZE + 1, PATCH_SIZE):

            patch = data[:, y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE]
            data_smooth = gaussian(patch, sigma=1, preserve_range=True)

            # -----------------------------
            # Intensity features
            # -----------------------------
            mean_intensity = float(np.mean(data_smooth))
            median_intensity = float(np.median(data_smooth))
            std_intensity = float(np.std(data_smooth))
            max_intensity = float(np.max(data_smooth))
            intensity_sum = float(np.sum(data_smooth))
            intensity_cv = float(std_intensity / (mean_intensity + EPS))
            p10_intensity = float(np.percentile(data_smooth, 10))
            p90_intensity = float(np.percentile(data_smooth, 90))

            # -----------------------------
            # Collagen density mask
            # -----------------------------
            try:
                threshold = threshold_otsu(data_smooth)
            except ValueError:
                threshold = np.mean(data_smooth)

            binary_mask = data_smooth > threshold
            collagen_density = float(np.sum(binary_mask) / binary_mask.size)

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

            coherence = (lambda1 - lambda2) / (lambda1 + lambda2 + lambda3 + EPS)
            mean_coherence = float(np.mean(coherence))
            std_coherence = float(np.std(coherence))

            # -----------------------------
            # Skeleton / morphology features
            # -----------------------------
            skeleton = skeletonize(binary_mask)

            endpoint_count, branchpoint_count, skeleton_voxels = compute_skeleton_graph_features(
                skeleton
            )

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
                    straightness = width / (length + EPS)

                    # convert length from pixels/voxels proxy to um approximately
                    # This is still a morphology proxy, not true fiber arc length.
                    fiber_lengths.append(length)
                    fiber_straightness.append(straightness)

                except Exception:
                    continue

            mean_length = float(np.mean(fiber_lengths)) if fiber_lengths else 0.0
            median_length = float(np.median(fiber_lengths)) if fiber_lengths else 0.0
            mean_straightness = float(np.mean(fiber_straightness)) if fiber_straightness else 0.0

            network_extent = float(skeleton_voxels / skeleton.size)

            # -----------------------------
            # Fiber diameter features
            # -----------------------------
            mean_fiber_diameter_um, median_fiber_diameter_um, std_fiber_diameter_um, p90_fiber_diameter_um = compute_fiber_diameter_features(
                binary_mask,
                skeleton,
                sampling=(Z_STEP_UM, PIXEL_SIZE_UM, PIXEL_SIZE_UM)
            )

            # -----------------------------
            # Component features
            # -----------------------------
            component_count, largest_component_fraction = compute_component_features(binary_mask)

            # -----------------------------
            # Spacing proxy features
            # -----------------------------
            mean_spacing_proxy_um, median_spacing_proxy_um = compute_spacing_features(
                binary_mask,
                sampling=(Z_STEP_UM, PIXEL_SIZE_UM, PIXEL_SIZE_UM)
            )

            # -----------------------------
            # 2D projection features
            # -----------------------------
            patch_proj = np.max(data_smooth, axis=0)
            mask_proj = np.max(binary_mask, axis=0)

            lacunarity_proxy = compute_lacunarity_proxy(mask_proj)

            dominant_orientation_deg, orientation_dispersion_deg = compute_orientation_features_from_projection(
                patch_proj
            )

            fft_mean_power, fft_radial_low_high_ratio, fft_anisotropy, fft_peak_orientation_deg = compute_fft_features(
                patch_proj
            )

            # -----------------------------
            # Store feature vector
            # -----------------------------
            feature_vector = {
                "file_name": name,
                "x": x0,
                "y": y0,

                # intensity
                "mean_intensity": mean_intensity,
                "median_intensity": median_intensity,
                "std_intensity": std_intensity,
                "max_intensity": max_intensity,
                "intensity_sum": intensity_sum,
                "intensity_cv": intensity_cv,
                "p10_intensity": p10_intensity,
                "p90_intensity": p90_intensity,

                # mask density
                "collagen_density": collagen_density,

                # tensor/coherence
                "mean_coherence": mean_coherence,
                "std_coherence": std_coherence,

                # skeleton/network
                "skeleton_voxels": skeleton_voxels,
                "network_extent": network_extent,
                "endpoint_count": endpoint_count,
                "branchpoint_count": branchpoint_count,
                "mean_fiber_length": mean_length,
                "median_fiber_length": median_length,
                "mean_fiber_straightness": mean_straightness,

                # diameter
                "mean_fiber_diameter_um": mean_fiber_diameter_um,
                "median_fiber_diameter_um": median_fiber_diameter_um,
                "std_fiber_diameter_um": std_fiber_diameter_um,
                "p90_fiber_diameter_um": p90_fiber_diameter_um,

                # connected components
                "component_count": component_count,
                "largest_component_fraction": largest_component_fraction,

                # spacing / porosity proxy
                "mean_spacing_proxy_um": mean_spacing_proxy_um,
                "median_spacing_proxy_um": median_spacing_proxy_um,

                # texture heterogeneity
                "lacunarity_proxy": lacunarity_proxy,

                # orientation
                "dominant_orientation_deg": dominant_orientation_deg,
                "orientation_dispersion_deg": orientation_dispersion_deg,

                # FFT
                "fft_mean_power": fft_mean_power,
                "fft_radial_low_high_ratio": fft_radial_low_high_ratio,
                "fft_anisotropy": fft_anisotropy,
                "fft_peak_orientation_deg": fft_peak_orientation_deg,
            }

            all_features.append(feature_vector)


# =========================================================
# Save CSV
# =========================================================
df = pd.DataFrame(all_features)

cols = ["file_name", "x", "y"] + [
    col for col in df.columns if col not in ["file_name", "x", "y"]
]

df = df[cols]

os.makedirs(save_path, exist_ok=True)

output_csv_path = os.path.join(
    save_path,
    f"{folder_name}_patch{PATCH_SIZE}_features_v2.csv"
)

df.to_csv(output_csv_path, index=False)

print(f"Features saved to: {output_csv_path}")
print("Done.")