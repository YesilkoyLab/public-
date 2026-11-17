import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Load CSV ---
df = pd.read_csv('../res/07172025-c1p1-zstack/07172025-c1p1-zstack_glcm_patch32_offset5.csv')

# --- Extract group ---
df['Group'] = df['file_name']

# --- Define color palette ---
group_order = ['1000-fw','1000-bw', '9010-fw','9010-bw', 
               '8020-fw', '8020-bw','7030-fw', '7030-bw',  
               '6040-fw', '6040-bw']
palette = dict(zip(group_order, sns.color_palette("tab20", len(group_order))))

# --- Feature categories ---
categories = ['xz_energy', 'xd_energy','yz_entropy', 'xd_entropy']

for category in categories:
    cols = [col for col in df.columns if category in col]
    plot_df = df[['Group'] + cols].melt(id_vars='Group', var_name='Direction', value_name='Value')

    plt.figure(figsize=(12, 6))
    directions = sorted(plot_df['Direction'].unique())
    group_count = len(group_order)

    # 每个 Direction 是一个主 x 位置
    for i, direction in enumerate(directions):
        for j, group in enumerate(group_order):
            sub_df = plot_df[(plot_df['Direction'] == direction) & (plot_df['Group'] == group)]

            # 计算该组在该方向的 x 偏移
            offset = (j - group_count/2) * 0.06

            # --- 右半边小提琴 ---
            sns.violinplot(
                data=sub_df,
                x=np.repeat(i + offset + 0.001, len(sub_df)),
                y='Value',
                color=palette[group],
                cut=0,
                bw=0.5,
                linewidth=0.6,
                inner=None,
                width=0.6
            )

            # --- 左半边散点 ---
            sns.stripplot(
                data=sub_df,
                x=np.repeat(i + offset - 0.001, len(sub_df)),
                y='Value',
                color=palette[group],
                size=2,
                jitter=True,
                alpha=0.6
            )

    plt.xticks(range(len(directions)), directions)
    plt.title(f'{category} - Half Violin (Right) + Scatter (Left)')
    plt.ylabel(category.split('_')[0].capitalize())
    plt.xlabel('Direction')
    plt.tight_layout()
    plt.savefig(f'{category}.png', dpi=300)
    plt.show()
