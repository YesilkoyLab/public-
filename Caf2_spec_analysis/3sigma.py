__author__ = 'Tianyi'

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.io import loadmat
from pdb import set_trace as st
from matplotlib import pyplot as plt
from glob import glob
from scipy.spatial import ConvexHull
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter, find_peaks


def rubberband_baseline_correction(x, y):
    """
    Rubberband baseline correction using the convex hull.
    
    Parameters:
        x (array-like): The x-axis values (e.g., wavenumber).
        y (array-like): The y-axis values (e.g., intensity).

    Returns:
        baseline (array): The rubberband baseline.
        corrected_y (array): The baseline-corrected spectrum.
    """
    x = np.array(x)
    y = np.array(y)

    # Get points forming the convex hull
    v = np.vstack((x, y)).T
    hull = ConvexHull(v)

    # Extract lower convex hull indices (start and end inclusive)
    hull_indices = sorted(hull.vertices)
    lower_indices = [idx for idx in hull_indices if idx == 0 or idx == len(x) - 1 or (y[idx] < y[idx-1] and y[idx] < y[idx+1])]
    lower_indices = np.array(sorted(lower_indices))

    # Interpolate baseline across those points
    baseline = np.interp(x, x[lower_indices], y[lower_indices])

    corrected_y = y - baseline

    return baseline, corrected_y


def main():

    foldername_list = ['kidney_oct'] #Caf2_03072025_rat_oct/liver_oct  #Caf2_03132025_rat_ffpe/liver_ffpe/
    filename_list = ['HMT_1'] #'HMT_10','HMT_5','HMT_4','HMT_3',
    cluster = ['cluster_0','cluster_1']
    #######load the spectrum after tsne filtering tsne_filter_save.py#########
    for foldername in foldername_list:
        for filename in filename_list:
            for cluster in cluster:
                print('processing:', foldername, filename, cluster)
                data = f'../res/rat/{foldername}/{filename}/{cluster}/spectra.npy'
                save_path = f'../res/rat/{foldername}/{filename}/figure'
                os.makedirs(save_path, exist_ok=True)

                wavelengths = np.linspace(950, 1800, 426)
                # data_after = loadmat(data)
                # spectra_after = np.reshape(data_after['r'], (480, 480, 426))
                spectrum = np.load(os.path.join(data))
                # st()
                spectra_after = np.reshape(spectrum, (100, 100, 426))

                # ----------------------------
                # 1. Average and Std
                # ----------------------------
                avg_spectrum = np.mean(spectrum, axis=0)
                std_spectrum = np.std(spectrum, axis=0)
                # st()
                _, corrected = rubberband_baseline_correction(wavelengths, avg_spectrum)
                avg_spectrum = corrected

                plt.figure(figsize=(8, 5))
                plt.plot(wavelengths, avg_spectrum, color="blue", label="Average Spectrum")
                plt.fill_between(
                    wavelengths,
                    avg_spectrum - std_spectrum,
                    avg_spectrum + std_spectrum,
                    color="blue",
                    alpha=0.3,
                    label="±1 STD",
                )
                plt.xlabel("Wavenumber (cm$^{-1}$)")
                plt.ylabel("Intensity (a.u.)")
                plt.title(f"Average Spectrum with STD Shading {cluster}")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(save_path, f"{cluster}_map.png"), dpi=300)
                plt.show()
                st()
                # ----------------------------
                # 2. 2nd Derivative Spectrum
                # ----------------------------
                # Savitzky-Golay filter for smooth derivative
                avg_spectrum_smooth = savgol_filter(avg_spectrum, window_length=11, polyorder=3)
                second_derivative = savgol_filter(avg_spectrum_smooth, window_length=11, polyorder=3, deriv=2)

                # ----------------------------
                # 3. Sigma in a Range
                # ----------------------------
                range_mask = (wavelengths >= 1150) & (wavelengths <= 1200)
                sigma_range = np.std(second_derivative[range_mask])
                threshold = 3 * sigma_range  # dips → positive peak in -2nd derivative

                # Use find_peaks on -second_derivative
                peaks_idx, properties = find_peaks(-second_derivative, height=threshold)
                dips_wavelengths = wavelengths[peaks_idx]

                print(f"Sigma in [950,1000]: {sigma_range:.4f}")
                print("Dips detected at:", dips_wavelengths)

                # ----------------------------
                # 4. Plot 2nd Derivative + Annotate Dips
                # ----------------------------
                plt.figure(figsize=(8, 5))
                plt.plot(wavelengths, second_derivative, color="red", label="2nd Derivative")
                plt.axhline(-threshold, color="gray", ls="--", label="3σ Threshold (dips)")

                # Mark dips
                for idx in peaks_idx:
                    wn = wavelengths[idx]
                    plt.scatter(wn, second_derivative[idx], color="black", s=30)
                    plt.text(wn+1, second_derivative[idx], f"{wn:.1f}", rotation=90, fontsize=8)

                plt.xlabel("Wavenumber (cm$^{-1}$)")
                plt.ylabel("2nd Derivative (a.u.)")
                plt.title("2nd Derivative Spectrum with 3σ Dip Detection")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(save_path, f"{cluster}_2nd_Derivative_Dips.png"), dpi=300)
                plt.show()



if __name__ == "__main__":
    main()