import os
import random
from glob import glob
from scipy.spatial import ConvexHull
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.io import loadmat
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler


def rubberband_correction(y, x):
    """Apply rubberband baseline correction to spectrum y with x-axis x."""
    

    points = np.column_stack((x, y))
    hull = ConvexHull(points)

    hull_indices = np.sort(hull.vertices)
    baseline_indices = [
        i for i in hull_indices
        if i == 0 or i == len(y) - 1 or (0 < i < len(y) - 1 and y[i] < y[i - 1] and y[i] < y[i + 1])
    ]

    if 0 not in baseline_indices:
        baseline_indices.insert(0, 0)
    if len(y) - 1 not in baseline_indices:
        baseline_indices.append(len(y) - 1)

    baseline_func = interp1d(
        x[baseline_indices],
        y[baseline_indices],
        kind="linear",
        fill_value="extrapolate"
    )
    baseline = baseline_func(x)

    corrected = y - baseline
    corrected[corrected < 0] = 0
    return corrected


def normalize_spectra_zscore(x):
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / (x.std() + 1e-8)


def extract_ratio_from_group(group_label):
    """
    Convert:
        1000_LMT_1 -> 1000
        9010SER_LMT_2 -> 9010SER
        6040PSER_LMT_3 -> 6040PSER
    """
    parts = group_label.split("_")
    if len(parts) < 3:
        return group_label
    return "_".join(parts[:-2])


def build_ratio_color_map():
    """
    Use dark color for SER and faded/lighter same-family color for PSER.
    Colors are taken from tab20 paired colors.
    """
    tab20 = plt.get_cmap("tab20").colors

    ratio_color_map = {
        "1000": tab20[14],      # gray-ish neutral
        "9010SER": tab20[0],    # dark blue
        "9010PSER": tab20[1],   # light blue
        "8020SER": tab20[2],    # dark orange
        "8020PSER": tab20[3],   # light orange
        "7030SER": tab20[4],    # dark green
        "7030PSER": tab20[5],   # light green
        "6040SER": tab20[6],    # dark red
        "6040PSER": tab20[7],   # light red
    }
    return ratio_color_map


def plot_avg_std_with_feature_markers(
    X,
    y_ratio,
    ratio_order,
    ratio_color_map,
    coef_importance,
    save_path,
    title,
    top_n=12,
    zoom_range=None,
):
    """
    Plot merged average curves by ratio + black feature marker lines at bottom.
    """
    wavelengths = np.linspace(950, 1800, X.shape[1])

    top_feature_indices = np.argsort(coef_importance)[::-1][:top_n]
    top_feature_indices = np.sort(top_feature_indices)

    if zoom_range is not None:
        start, end = zoom_range
        valid_idx = np.where((wavelengths >= start) & (wavelengths <= end))[0]
    else:
        valid_idx = np.arange(len(wavelengths))

    plt.figure(figsize=(12, 5))

    for ratio in ratio_order:
        group_data = X[y_ratio == ratio]
        if len(group_data) == 0:
            continue

        avg = np.mean(group_data, axis=0)
        std = np.std(group_data, axis=0)

        plt.plot(
            wavelengths[valid_idx],
            avg[valid_idx],
            label=ratio,
            color=ratio_color_map[ratio],
            linewidth=2.5
        )
        plt.fill_between(
            wavelengths[valid_idx],
            avg[valid_idx] - std[valid_idx],
            avg[valid_idx] + std[valid_idx],
            color=ratio_color_map[ratio],
            alpha=0.18
        )

    ax = plt.gca()
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min

    new_ymin = y_min - 0.16 * y_range
    ax.set_ylim(new_ymin, y_max)

    marker_y0 = y_min - 0.12 * y_range
    marker_y1 = y_min - 0.04 * y_range

    top_feature_indices_in_range = [
        idx for idx in top_feature_indices
        if wavelengths[idx] >= wavelengths[valid_idx].min()
        and wavelengths[idx] <= wavelengths[valid_idx].max()
    ]

    for idx in top_feature_indices_in_range:
        wx = wavelengths[idx]
        ax.plot(
            [wx, wx],
            [marker_y0, marker_y1],
            color="black",
            linewidth=2,
            solid_capstyle="butt"
        )

    plt.title("Second derivative spectrum with top LDA features",fontsize=18)
    plt.xlabel("Wavenumber (cm$^{-1}$)",fontsize=16)
    plt.ylabel("Normalized Intensity (a.u.)",fontsize=16)
    plt.yticks([])
    # plt.legend(frameon=False, loc="upper left")
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_path, f"{title.replace(' ', '_')}.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    feature_table = pd.DataFrame({
        "feature_index": top_feature_indices,
        "wavenumber": wavelengths[top_feature_indices],
        "importance": coef_importance[top_feature_indices]
    })
    feature_table.to_csv(
        os.path.join(save_path, f"{title.replace(' ', '_')}_top_features.csv"),
        index=False
    )


def main():
    random.seed(42)
    np.random.seed(42)

    name = "Caf2_03272026"
    spectral_root = r"../../spec_analysis/res/Caf2_03272026/2nd"
    save_path = os.path.join(spectral_root, "lda_result_ratio_overlay")
    os.makedirs(save_path, exist_ok=True)

    # 9 samples
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

    subfolders = ["LMT_1"] #, "LMT_2", "LMT_3"

    # dark color for SER, faded same color for PSER
    ratio_color_map = build_ratio_color_map()

    # number sampled from each subfolder
    sample_per_group = 256
    expected_length = 426
    wavelengths = np.linspace(950, 1800, expected_length)

    # preprocessing options
    apply_rubberband = False
    apply_area_normalization = False
    apply_zscore = False
    read_mat_instead_of_npy = False

    # collect spectral data for 27-group LDA (9 ratios x 3 LMT)
    spectral_data = []
    labels_group = []
    file_records = []

    for ratio in ratios:
        for subfolder in subfolders:
            folder_path = os.path.join(spectral_root, ratio, subfolder)

            if read_mat_instead_of_npy:
                all_files = sorted(glob(os.path.join(folder_path, "spectrum_*.mat")))
            else:
                all_files = sorted(glob(os.path.join(folder_path, "*.npy")))

            group_name = f"{ratio}_{subfolder}"

            if len(all_files) == 0:
                print(f"Warning: no files found for {group_name}")
                continue

            if len(all_files) >= sample_per_group:
                sampled_files = random.sample(all_files, sample_per_group)
            else:
                print(f"Warning: {group_name} only has {len(all_files)} files.")
                sampled_files = all_files

            valid_count = 0
            for f in sampled_files:
                try:
                    if read_mat_instead_of_npy:
                        data = loadmat(f)
                        spectrum = data["spectrum"].flatten()
                    else:
                        spectrum = np.load(f)

                    spectrum = np.asarray(spectrum).flatten()

                    if len(spectrum) != expected_length:
                        print(f"Skipping {f}: expected length {expected_length}, got {len(spectrum)}")
                        continue

                    if np.any(~np.isfinite(spectrum)):
                        print(f"Skipping {f}: contains NaN or inf")
                        continue

                    if apply_rubberband:
                        spectrum = rubberband_correction(spectrum, wavelengths)

                    if apply_area_normalization:
                        area = np.trapz(spectrum, x=wavelengths)
                        if area > 0:
                            spectrum = spectrum / area

                    if apply_zscore:
                        spectrum = normalize_spectra_zscore(spectrum)

                    spectral_data.append(spectrum)
                    labels_group.append(group_name)
                    file_records.append({
                        "file": f,
                        "ratio": ratio,
                        "subfolder": subfolder,
                        "group_label": group_name
                    })
                    valid_count += 1

                except Exception as e:
                    print(f"Failed to load {f}: {e}")

            print(f"{group_name}: {valid_count} spectra loaded.")

    if len(spectral_data) == 0:
        raise ValueError("No valid spectra were loaded.")

    spectral_df = pd.DataFrame(spectral_data)
    spectral_df["label_group"] = labels_group

    X = spectral_df.drop(columns=["label_group"])
    y_group = spectral_df["label_group"].values
    X.columns = X.columns.astype(str)

    # merged 9-class label for plotting average spectra
    y_ratio = np.array([extract_ratio_from_group(g) for g in y_group])

    # standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # LDA on all groups
    n_classes = len(np.unique(y_group))
    n_features = X_scaled.shape[1]
    n_components = min(2, n_features, n_classes - 1)

    if n_components < 1:
        raise ValueError("Need at least 2 classes for LDA.")

    lda = LinearDiscriminantAnalysis(n_components=n_components)
    X_lda = lda.fit_transform(X_scaled, y_group)

    # 27-group order
    group_order = [f"{ratio}_{subfolder}" for ratio in ratios for subfolder in subfolders]

    # use same color family for all LMTs of same ratio
    group_color_map = {}
    for ratio in ratios:
        base_color = ratio_color_map[ratio]
        for subfolder in subfolders:
            group_color_map[f"{ratio}_{subfolder}"] = base_color

    # LDA scatter plot
    if n_components == 1:
        lda_df = pd.DataFrame({
            "LDA1": X_lda[:, 0],
            "Group": y_group
        })

        plt.figure(figsize=(10, 8))
        for i, group in enumerate(group_order):
            mask = lda_df["Group"] == group
            vals = lda_df.loc[mask, "LDA1"].values
            if len(vals) == 0:
                continue

            jitter = np.random.normal(i, 0.06, size=len(vals))
            plt.scatter(
                vals,
                jitter,
                alpha=0.70,
                s=28,
                label=group,
                color=group_color_map[group],
                edgecolors="none"
            )

        plt.yticks(range(len(group_order)), group_order)
        plt.xlabel("LDA Component 1",fontsize=20)
        plt.ylabel("Group",fontsize=20)
        plt.title("LDA: Spectral Features",fontsize=20)
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, ncol=1)
        plt.tight_layout()
        plt.savefig(
            os.path.join(save_path, f"({name})_LDA_scatter_groups.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

    else:
        lda_df = pd.DataFrame({
            "LDA1": X_lda[:, 0],
            "LDA2": X_lda[:, 1],
            "Group": y_group
        })

        plt.figure(figsize=(10, 8))
        for group in group_order:
            mask = lda_df["Group"] == group
            if np.sum(mask) == 0:
                continue

            plt.scatter(
                lda_df.loc[mask, "LDA1"],
                lda_df.loc[mask, "LDA2"],
                alpha=0.70,
                s=28,
                label=group,
                color=group_color_map[group],
                edgecolors="none"
            )

        plt.xlabel("LDA Component 1", fontsize=20)
        plt.ylabel("LDA Component 2", fontsize=20)
        plt.title("LDA: Spectral Features", fontsize=20)
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, ncol=1)
        plt.tight_layout()
        plt.savefig(
            os.path.join(save_path, f"({name})_LDA_scatter_groups.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

    # feature importance from LDA
    num_features = X.shape[1]
    wavenumbers = np.linspace(950, 1800, num=num_features)
    feature_names = [f"{int(round(w))} cm⁻¹" for w in wavenumbers]

    coef_importance = np.sum(np.abs(lda.coef_), axis=0)

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Wavenumber": wavenumbers,
        "Importance": coef_importance
    }).sort_values(by="Importance", ascending=False)

    importance_df.to_csv(
        os.path.join(save_path, f"({name})_LDA_feature_importance.csv"),
        index=False
    )

    # merged 9 curves + black feature lines
    plot_avg_std_with_feature_markers(
        X=X.to_numpy(),
        y_ratio=y_ratio,
        ratio_order=ratios,
        ratio_color_map=ratio_color_map,
        coef_importance=coef_importance,
        save_path=save_path,
        title="Average Spectrum with Top LDA Features",
        top_n=40,
        zoom_range=(950, 1800),
    )

    # optional: save sampled file log
    # pd.DataFrame(file_records).to_csv(
    #     os.path.join(save_path, f"({name})_sampled_files.csv"),
    #     index=False
    # )

    print("\nDone.")
    print(f"Results saved to: {save_path}")
    print(f"All group labels: {np.unique(y_group)}")
    print(f"9 merged ratios: {np.unique(y_ratio)}")
    print(f"LDA n_components: {n_components}")
    print(f"Input shape: {X.shape}")
    print(f"LDA output shape: {X_lda.shape}")
    print("\nMain output figure:")
    print("  Average Spectrum with Top LDA Features (Merged by Ratio).png")


if __name__ == "__main__":
    main()