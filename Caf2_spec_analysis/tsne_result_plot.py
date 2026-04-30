import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle

# ---------------------- #
# Data Loading Functions #
# ---------------------- #
def load_npy_data(folder, max_samples=1000):
    print(f"Processing folder: {folder}")
    files = sorted([f for f in os.listdir(folder) if f.endswith('.npy')])
    if len(files) == 0:
        print(f"⚠️ Nothing in this folder: {folder}")
        return np.array([])
    np.random.seed(42)
    files = np.random.choice(files, size=min(max_samples, len(files)), replace=False)
    data = []
    for file in files:
        arr = np.load(os.path.join(folder, file))
        if np.any(arr):
            data.append(arr.flatten())
    return np.array(data)

def normalize_spectra_zscore(X):
    X_mean = X.mean(axis=1, keepdims=True)
    X_std = X.std(axis=1, keepdims=True)
    return (X - X_mean) / (X_std + 1e-8)

# -------------------------- #
# Plotting Function: Avg/STD #
# -------------------------- #
def plot_avg_std_with_top_features(X, y, label_map, top_feature_indices, top_importance, save_path, title, zoom_ranges=None):
    wavelengths = np.linspace(950, 1800, X.shape[1])
    unique_labels = np.unique(y)

    # ✅ FIX: folder colors dictionary
    folder_colors = {
        'liver_ffpe': 'orange',
        'kidney_ffpe': 'blue',
        'liver_ff': 'green',
        'kidney_ff': 'purple'
    }

    plt.figure(figsize=(14, 8))
    for label in unique_labels:
        group_data = X[y == label]
        if group_data.size == 0:
            continue

        avg = np.mean(group_data, axis=0)
        std = np.std(group_data, axis=0)

        # ✅ FIX: recover folder name directly from mapping
        folder_name = label_map[label]
        color = folder_colors.get(folder_name, 'gray')

        plt.plot(wavelengths, avg, label=folder_name, color=color, linewidth=3)
        plt.fill_between(wavelengths, avg - std, avg + std, color=color, alpha=0.2)

    # draw top features
    y_min, y_max = plt.ylim()
    tick_height = 0.05 * (y_max - y_min)
    for idx in top_feature_indices:
        wn = wavelengths[idx]
        plt.vlines(wn, y_min, y_min + tick_height, color='black', linewidth=1.5)

    plt.title(f"{title}")
    plt.xlabel("Wavenumber (cm$^{-1}$)", fontsize=14)
    plt.ylabel("Normalized Intensity", fontsize=14)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=24, loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{title.replace(' ', '_')}_full_spectrum.png"))
    plt.show()
    plt.close()

# ---------------------- #
# Main Analysis Pipeline #
# ---------------------- #
def main():
    foldername_list = ['kidney_ffpe', 'kidney_ff'] #'kidney_ffpe',
    filename_list = ['HMT_1']

    base_path = "D:/spec_res/rat"
    save_path = "D:/spec_res/rat/result/"
    os.makedirs(save_path, exist_ok=True)

    all_data = []
    all_labels = []
    label_map = {}  # ✅ FIX: map numeric label -> folder name
    label_index = 0

    for foldername in foldername_list:
        for filename in filename_list:
            folder_path = os.path.join(base_path, foldername, filename)
            if not os.path.isdir(folder_path):
                continue
            data = load_npy_data(folder_path, max_samples=100)
            if len(data) == 0:
                continue
            norm_data = normalize_spectra_zscore(data)
            all_data.append(norm_data)
            all_labels += [label_index] * len(norm_data)
            label_map[label_index] = foldername  # ✅ FIX: link label to folder name
            print(f"Loaded {len(norm_data)} data from {foldername}/{filename} as label {label_index}")
            label_index += 1

    if len(all_data) == 0:
        print("No data loaded. Exiting.")
        return

    X = np.concatenate(all_data, axis=0)
    y = np.array(all_labels)
    X_small, y_small = shuffle(X, y, random_state=42)
    X_small = X_small[:1000]
    y_small = y_small[:1000]

    # --------------------------- #
    # Feature Importance (LogReg) #
    # --------------------------- #
    clf = LogisticRegression(penalty='l2', solver='liblinear', max_iter=1000)
    clf.fit(X_small, y_small)
    importance = np.mean(np.abs(clf.coef_), axis=0)
    wavenumbers = np.linspace(950, 1800, len(importance))
    top_indices = np.argsort(importance)[-40:]
    top_importance = importance[top_indices]

    # ------------------------ #
    # Avg ± STD Spectrum Plot #
    # ------------------------ #
    plot_avg_std_with_top_features(
        X_small, y_small, label_map,
        top_indices, top_importance,
        save_path,
        title="Avg STD Spectrum with Top Features"
    )

if __name__ == '__main__':
    main()
