import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import numpy as np


# =========================================================
# Settings
# =========================================================
glcm_csv_paths = [
    r"../res-15/03252026/03252026_glcm_patch32_offset5_15slice.csv",
    r"../res-15/03272026/03272026_glcm_patch32_offset5_15slice.csv",
    r"../res-15/04202026/04202026_glcm_patch32_offset5_15slice.csv",
]

labels = ["replicate1", "replicate2", "replicate3"]

marker_map = {
    "replicate1": ".",
    "replicate2": "*",
    "replicate3": "o"
}

x_order = [
    "1000",
    "9010ser", "8020ser", "7030ser", "6040ser",
    "9010pser", "8020pser", "7030pser", "6040pser"
]

save_dir = r"../res-15/three_csv_compare"
os.makedirs(save_dir, exist_ok=True)

sns.set_style("white")

# all GLCM features you want
glcm_features = [
    "a0_contrast"# , "a0_energy", "a0_entropy",
    # "a1_contrast", "a1_energy", "a1_entropy",
    # "a2_contrast", "a2_energy", "a2_entropy",
    # "a0a1_contrast", "a0a1_energy", "a0a1_entropy",
    # "a1a2_contrast", "a1a2_energy", "a1a2_entropy",
    # "a0a2_contrast", "a0a2_energy", "a0a2_entropy",
    # "d111_contrast", "d111_energy", "d111_entropy",
]


# =========================================================
# Helpers
# =========================================================
def minmax_normalize(series):
    smin = series.min()
    smax = series.max()
    if pd.isna(smin) or pd.isna(smax) or smax == smin:
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - smin) / (smax - smin)


def extract_base_name(fname, x_order):
    if not isinstance(fname, str):
        return None

    fname = fname.lower().strip()
    base = re.sub(r'_\d+$', '', fname)

    valid = [x.lower() for x in x_order]
    if base in valid:
        return base
    return None


def standardize_file_name(df):
    if "file_name" not in df.columns:
        lower_map = {c.lower(): c for c in df.columns}
        if "file_name" in lower_map:
            df = df.rename(columns={lower_map["file_name"]: "file_name"})
        else:
            raise ValueError(f"'file_name' column not found. Available columns: {list(df.columns)}")

    df["file_name"] = df["file_name"].astype(str).str.strip().str.lower()
    return df


def find_exact_col(df, name):
    lower_map = {c.lower(): c for c in df.columns}
    if name.lower() in lower_map:
        return lower_map[name.lower()]
    raise ValueError(f"Column '{name}' not found. Available columns: {list(df.columns)}")


def build_ratio_color_map():
    tab20 = plt.get_cmap("tab20").colors
    return {
        "1000": tab20[14],
        "9010ser": tab20[0],
        "9010pser": tab20[1],
        "8020ser": tab20[2],
        "8020pser": tab20[3],
        "7030ser": tab20[4],
        "7030pser": tab20[5],
        "6040ser": tab20[6],
        "6040pser": tab20[7],
    }


# =========================================================
# Data Processing
# =========================================================
def process_glcm(path, label, x_order, glcm_features):
    df = pd.read_csv(path)
    df = standardize_file_name(df)

    out_dict = {"file_name": df["file_name"]}

    for feat in glcm_features:
        try:
            col = find_exact_col(df, feat)
            out_dict[feat] = df[col]
        except Exception:
            print(f"Warning: '{feat}' not found in {path}")

    out = pd.DataFrame(out_dict)

    out["dataset"] = label
    out["base_name"] = out["file_name"].apply(lambda x: extract_base_name(x, x_order))
    out = out[out["base_name"].notna()].copy()

    return out


# =========================================================
# Plot
# =========================================================
def plot_metric_multi_csv(data, y_col, title, ylabel, color_map, x_order, save_dir, marker_map):
    plt.figure(figsize=(7, 3))

    datasets = list(marker_map.keys())
    n_datasets = len(datasets)

    offsets = np.linspace(-0.18, 0.18, n_datasets)
    offset_map = {ds: off for ds, off in zip(datasets, offsets)}

    def get_marker_size(marker):
        if marker == ".":
            return 20
        elif marker == "*":
            return 55
        elif marker == "o":
            return 24
        else:
            return 28

    for ds in datasets:
        sub = data[data["dataset"] == ds].copy()

        if sub.empty:
            continue

        x_numeric = sub["base_name"].cat.codes.astype(float) + offset_map[ds]
        point_colors = sub["base_name"].astype(str).map(color_map).tolist()

        plt.scatter(
            x_numeric,
            sub[y_col],
            c=point_colors,
            s=get_marker_size(marker_map[ds]),
            marker=marker_map[ds],
            alpha=0.75,
            edgecolors="gray",
            linewidths=0.25
        )

    mean_df = (
        data.groupby(["dataset", "base_name"], observed=False)[y_col]
        .mean()
        .reset_index()
    )

    line_halfwidth = 0.05
    for _, row in mean_df.iterrows():
        ds = row["dataset"]
        base = row["base_name"]
        yv = row[y_col]

        if pd.isna(base) or pd.isna(yv):
            continue

        i = x_order.index(str(base))
        xc = i + offset_map[ds]

        plt.plot(
            [xc - line_halfwidth, xc + line_halfwidth],
            [yv, yv],
            color="black",
            lw=3
        )

    plt.title(title, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.xlabel("")
    plt.xticks(range(len(x_order)), x_order, rotation=15, fontsize=16)
    plt.yticks([])

    legend_handles = [
        Line2D(
            [0], [0],
            marker=marker_map[ds],
            color="black",
            linestyle="None",
            markersize=6 if marker_map[ds] == "." else 8,
            label=ds
        )
        for ds in datasets
    ]

    plt.legend(
        handles=legend_handles,
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        title="Replicate"
    )

    plt.tight_layout()

    safe_title = title.replace(" ", "_").replace("/", "_")
    plt.savefig(os.path.join(save_dir, f"{safe_title}.png"), dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


# =========================================================
# Main
# =========================================================
grouped_dfs = [
    process_glcm(p, l, x_order, glcm_features)
    for p, l in zip(glcm_csv_paths, labels)
]

all_data = pd.concat(grouped_dfs, ignore_index=True)

all_data["base_name"] = pd.Categorical(
    all_data["base_name"],
    categories=x_order,
    ordered=True
)

metrics = glcm_features

# normalize within each csv separately
for m in metrics:
    if m in all_data.columns:
        all_data[f"{m}_norm"] = (
            all_data.groupby("dataset")[m]
            .transform(minmax_normalize)
        )
    else:
        print(f"Skipping normalization for missing column: {m}")

color_map = build_ratio_color_map()


# =========================================================
# Plots
# =========================================================
for m in metrics:
    norm_col = f"{m}_norm"
    if norm_col not in all_data.columns:
        print(f"Skipping plot for missing normalized column: {norm_col}")
        continue

    plot_metric_multi_csv(
        all_data,
        norm_col,
        m.replace("_", " ").title(),
        "Normalized value",
        color_map,
        x_order,
        save_dir,
        marker_map
    )