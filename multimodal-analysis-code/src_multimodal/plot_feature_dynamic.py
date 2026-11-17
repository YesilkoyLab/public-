import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =========================================================
# Settings
# =========================================================
glcm_csv_paths = [
    r"../res-15/03252026/03252026_glcm_patch32_offset5_15slice.csv"
]

labels = ["03252026"]

x_order = [
    "1000",
    "9010ser", "8020ser", "7030ser", "6040ser",
    "9010pser", "8020pser", "7030pser", "6040pser"
]

save_dir = r"../res-15/03252026/figures-glcm-only"
os.makedirs(save_dir, exist_ok=True)

sns.set_style("white")


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
def process_glcm(path, label, x_order):
    df = pd.read_csv(path)
    df = standardize_file_name(df)

    a0_contrast_col = find_exact_col(df, "a0_contrast")
    a2_entropy_col = find_exact_col(df, "a1_entropy")
    a2_energy_col = find_exact_col(df, "a2_energy")

    out = pd.DataFrame({
        "file_name": df["file_name"],
        "a0_contrast": df[a0_contrast_col],
        "a1_entropy": df[a2_entropy_col],
        "a2_energy": df[a2_energy_col],
    })

    out["dataset"] = label
    out["base_name"] = out["file_name"].apply(lambda x: extract_base_name(x, x_order))
    out = out[out["base_name"].notna()].copy()

    return out


# =========================================================
# Plot
# =========================================================
def plot_metric(data, y_col, title, ylabel, color_map, x_order, save_dir):
    plt.figure(figsize=(9, 2.8))

    ax = sns.stripplot(
        data=data,
        x="base_name",
        y=y_col,
        hue="base_name",
        order=x_order,
        palette=color_map,
        size=2.8,
        jitter=0.18,
        alpha=0.7,
        edgecolor="gray",
        linewidth=0.25
    )

    base_avg = data.groupby("base_name", observed=False)[y_col].mean()
    for i, base in enumerate(x_order):
        if base in base_avg.index and pd.notna(base_avg.loc[base]):
            plt.plot(
                [i - 0.18, i + 0.18],
                [base_avg.loc[base], base_avg.loc[base]],
                color="black",
                lw=3.5
            )

    plt.title(title, fontsize=12)
    plt.ylabel(ylabel, fontsize=10)
    plt.xlabel("")
    plt.xticks(rotation=0, fontsize=9)
    plt.yticks([])

    leg = ax.get_legend()
    if leg is not None:
        leg.remove()

    plt.tight_layout()

    safe_title = title.replace(" ", "_").replace("/", "_")
    plt.savefig(os.path.join(save_dir, f"{safe_title}.png"), dpi=300, bbox_inches="tight")
    plt.show()


# =========================================================
# Main
# =========================================================
grouped_dfs = [
    process_glcm(p, l, x_order)
    for p, l in zip(glcm_csv_paths, labels)
]

all_data = pd.concat(grouped_dfs, ignore_index=True)

all_data["base_name"] = pd.Categorical(
    all_data["base_name"],
    categories=x_order,
    ordered=True
)

metrics = [
    "a0_contrast",
    "a2_energy",
    "a1_entropy"
]

for m in metrics:
    all_data[f"{m}_norm"] = minmax_normalize(all_data[m])

color_map = build_ratio_color_map()


# =========================================================
# Plots
# =========================================================
plot_metric(
    all_data,
    "a0_contrast_norm",
    "A0 Contrast",
    "Normalized value",
    color_map,
    x_order,
    save_dir
)

plot_metric(
    all_data,
    "a1_entropy_norm",
    "A1 Entropy",
    "Normalized value",
    color_map,
    x_order,
    save_dir
)

plot_metric(
    all_data,
    "a2_energy_norm",
    "A2 Energy",
    "Normalized value",
    color_map,
    x_order,
    save_dir
)