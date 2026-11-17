import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# Settings
# =========================================================
np.random.seed(7)

output_path = "feature_importance_heatmap_estimated_more_std.png"

samples = [
    "1000",
    "6040PSER",
    "6040SER",
    "7030PSER",
    "7030SER",
    "8020PSER",
    "8020SER",
    "9010PSER",
    "9010SER",
]

# spectral features
wavenumbers = np.arange(950, 1701, 4)

# add some SHG-like features on the right, like your figure
extra_features = [
    "a2 energy",
    "a0 energy",
    "a1 contrast",
    "a0 contrast",
    "a2 entropy",
    "d111 entropy",
    "median intensity",
    "fiber coherence",
]

feature_names = [str(w) for w in wavenumbers] + extra_features

# =========================================================
# Helper
# =========================================================
def gaussian(x, mu, sigma, amp):
    return amp * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

# =========================================================
# Build estimated feature importance matrix
# Goal:
# - Amide I remains strongest
# - 1044 and 1086 lower than Amide I
# - More std / more realistic fluctuation
# =========================================================
data = []

for i, sample in enumerate(samples):
    x = wavenumbers.astype(float)

    # baseline around zero
    row = np.random.normal(0, 18, len(x))

    # smooth low-frequency structure
    row += 20 * np.sin((x - 900) / 95 + i * 0.35)
    row += 12 * np.cos((x - 900) / 55 + i * 0.20)

    # main spectral regions
    row += gaussian(x, 1044, 10, 110 + np.random.normal(0, 8))
    row += gaussian(x, 1086, 11, 95 + np.random.normal(0, 8))
    row += gaussian(x, 1450, 18, 60 + np.random.normal(0, 6))
    row += gaussian(x, 1545, 16, 90 + np.random.normal(0, 6))
    row += gaussian(x, 1655, 18, 180 + np.random.normal(0, 8))   # amide I strongest

    # negative dips
    row -= gaussian(x, 1200, 16, 40 + np.random.normal(0, 5))
    row -= gaussian(x, 1380, 20, 30 + np.random.normal(0, 5))
    row -= gaussian(x, 1590, 14, 35 + np.random.normal(0, 5))

    # sample-dependent variation
    if "PSER" in sample:
        row += gaussian(x, 1044, 9, 10)
        row += gaussian(x, 1086, 10, 8)
        row += gaussian(x, 1240, 16, 18)
    elif "SER" in sample and sample != "1000":
        row += gaussian(x, 1036, 9, 6)
        row += gaussian(x, 1452, 18, 10)

    # more std / variability
    raw_noise = np.random.normal(0, 1, len(x))
    kernel = np.exp(-np.linspace(-2.5, 2.5, 17) ** 2)
    kernel = kernel / kernel.sum()
    corr_noise = np.convolve(raw_noise, kernel, mode="same")

    region_weight = (
        1.0
        + 0.8 * np.exp(-(x - 1050) ** 2 / (2 * 70 ** 2))
        + 0.9 * np.exp(-(x - 1650) ** 2 / (2 * 80 ** 2))
    )

    row += corr_noise * 38 * region_weight

    # local stripe-like spikes
    spike_idx = np.random.choice(len(x), size=18, replace=False)
    row[spike_idx] += np.random.normal(0, 55, size=len(spike_idx))

    # additional independent noise
    row += np.random.normal(0, 16, len(x))

    data.append(row)

data = np.array(data)

# =========================================================
# Add SHG-like feature columns on the right
# =========================================================
extra_data = []
for i, sample in enumerate(samples):
    vals = np.array([
        np.random.normal(40, 35),
        np.random.normal(25, 30),
        np.random.normal(10, 28),
        np.random.normal(-35, 30),
        np.random.normal(-15, 25),
        np.random.normal(-5, 20),
        np.random.normal(55, 35),
        np.random.normal(20, 30),
    ])

    if "PSER" in sample:
        vals += np.array([8, 6, 0, -6, -4, 0, 10, 5])
    elif "SER" in sample and sample != "1000":
        vals += np.array([4, 3, 0, -3, -2, 0, 5, 3])

    extra_data.append(vals)

extra_data = np.array(extra_data)

full_data = np.hstack([data, extra_data])
heatmap_df = pd.DataFrame(full_data, index=samples, columns=feature_names)

# =========================================================
# Plot
# =========================================================
plt.figure(figsize=(16, 4.8))

ax = sns.heatmap(
    heatmap_df,
    cmap="RdBu_r",
    center=0,
    vmin=-400,
    vmax=400,
    xticklabels=False,   # 先关掉默认全部显示
    yticklabels=True,
    cbar_kws={"label": "LDA Coefficient"}
)

ax.set_title("Feature Importance Heatmap", fontsize=16)
ax.set_xlabel("")
ax.set_ylabel("")

# =========================================================
# Custom x ticks:
# spectral part -> show every 100 cm^-1
# extra features -> show all
# =========================================================
spectral_tick_positions = []
spectral_tick_labels = []

for target in range(1000, 1801, 100):
    idx = np.argmin(np.abs(wavenumbers - target))
    spectral_tick_positions.append(idx + 0.5)
    spectral_tick_labels.append(str(int(wavenumbers[idx])))

extra_tick_positions = []
extra_tick_labels = []

offset = len(wavenumbers)
for j, feat in enumerate(extra_features):
    extra_tick_positions.append(offset + j + 0.5)
    extra_tick_labels.append(feat)

all_tick_positions = spectral_tick_positions + extra_tick_positions
all_tick_labels = spectral_tick_labels + extra_tick_labels

ax.set_xticks(all_tick_positions)
ax.set_xticklabels(all_tick_labels, rotation=60, ha="right", fontsize=10)

plt.yticks(rotation=0, fontsize=8)

plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved to: {output_path}")