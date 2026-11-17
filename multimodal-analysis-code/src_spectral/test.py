import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# =========================
# User settings
# =========================
corrected_spectra_path = r"/Users/tianyizheng/Documents/tzws/spec_analysis/res/rat/liver_ffpe/HMT_3_corrected_spectra_all.npy"
cluster_labels_path = r"/Users/tianyizheng/Documents/tzws/spec_analysis/res/rat/liver_ffpe/HMT_3_cluster_labels.npy"

save_path = r"/Users/tianyizheng/Documents/tzws/spec_analysis/res/rat/test"
filename = "HMT_3"

selected_k = 4
smooth = True
savgol_window = 11   # must be odd
savgol_poly = 2

same_ylim = True
ylim = (0, 1.3)

cluster_colors = {
    1: "#1f77b4",  # blue
    2: "#d62728",  # red
    3: "#e377c2",  # magenta/pink
    4: "#17becf",  # cyan
}

# =========================
# Load data
# =========================
os.makedirs(save_path, exist_ok=True)

corrected_spectra = np.load(corrected_spectra_path)
cluster_labels_1based = np.load(cluster_labels_path)
wavelengths = np.linspace(950, 1800, 426)

corrected_spectra = np.asarray(corrected_spectra)
cluster_labels_1based = np.asarray(cluster_labels_1based).ravel()
wavelengths = np.asarray(wavelengths).ravel()

if corrected_spectra.shape[0] != cluster_labels_1based.shape[0]:
    raise ValueError(
        f"Number of spectra ({corrected_spectra.shape[0]}) does not match "
        f"number of labels ({cluster_labels_1based.shape[0]})."
    )

if corrected_spectra.shape[1] != wavelengths.shape[0]:
    raise ValueError(
        f"Spectrum length ({corrected_spectra.shape[1]}) does not match "
        f"number of wavelengths ({wavelengths.shape[0]})."
    )

# =========================
# Plot average spectra per cluster
# =========================
plt.figure(figsize=(7, 5))

for c in range(1, selected_k + 1):
    cluster_spectra = corrected_spectra[cluster_labels_1based == c]

    if cluster_spectra.shape[0] == 0:
        print(f"Warning: Cluster {c} has no spectra. Skipping.")
        continue

    mean_spec = np.mean(cluster_spectra, axis=0)
    std_spec = np.std(cluster_spectra, axis=0)

    if smooth:
        mean_plot = savgol_filter(mean_spec, savgol_window, savgol_poly)
        std_plot = savgol_filter(std_spec, savgol_window, savgol_poly)
    else:
        mean_plot = mean_spec
        std_plot = std_spec

    color = cluster_colors.get(c, None)

    plt.plot(
        wavelengths,
        mean_plot,
        color=color,
        linewidth=2,
        label=f"Cluster {c}, n={cluster_spectra.shape[0]}"
    )

    plt.fill_between(
        wavelengths,
        mean_plot - std_plot,
        mean_plot + std_plot,
        color=color,
        alpha=0.15,
        linewidth=0
    )

plt.xlabel(r"Wavenumber cm$^{-1}$")
plt.ylabel("Baseline-corrected intensity")
plt.title(f"{filename} mean spectra by cluster")
plt.legend(fontsize=12)

if same_ylim:
    plt.ylim(*ylim)

plt.tight_layout()

out_png = os.path.join(save_path, f"{filename}_cluster_mean_spectra_cluster_colors.png")
plt.savefig(out_png, dpi=300)
plt.close()

print(f"Saved: {out_png}")