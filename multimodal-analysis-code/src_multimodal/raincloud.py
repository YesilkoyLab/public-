import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Load data ---
# df = pd.read_csv('../res/09022025-pser-zstack/09022025-pser-zstack_glcm_patch32_offset5.csv')
df = pd.read_csv('../res/09022025-pser-zstack/09022025-pser-zstack_patch_features.csv')
df['Group'] = df['file_name']

# --- Groups and palette ---
# group_order = ['1000-fw', '9010-fw',
#                '8020-fw', '7030-fw',
#                '6040-fw']
group_order = ['1000-fw-1','1000-bw-1', '9010-fw-1','9010-bw-1',
               '8020-fw-1', '8020-bw-1','7030-fw-1', '7030-bw-1',
               '6040-fw-1', '6040-bw-1']
palette = dict(zip(group_order, sns.color_palette("tab20", len(group_order))))

# categories = ['xz_energy','yd_contrast', 'yz_entropy','dd_homogeneity']
categories = ['mean_fiber_straightness','mean_fiber_length']
sns.set(style="whitegrid")

# --- Spacing parameters ---
group_spacing = 0.35   # horizontal space between groups within a direction
violin_width = 0.3
box_width = 0.1 

for category in categories:
    cols = [col for col in df.columns if category in col]
    plot_df = df[['Group'] + cols].melt(id_vars='Group', var_name='Direction', value_name='Value')
    directions = sorted(plot_df['Direction'].unique())

    fig, ax = plt.subplots(figsize=(10, 6))

    for dir_index, direction in enumerate(directions):
        for grp_index, group in enumerate(group_order):
            subset = plot_df[(plot_df['Direction'] == direction) & (plot_df['Group'] == group)]
            values = subset['Value'].values

            # Calculate manual position for this group in this direction
            pos = dir_index + (grp_index - (len(group_order) - 1) / 2) * group_spacing

            violin_pos = pos #+ box_width / 50
            # --- Violin ---
            parts = ax.violinplot(
                dataset=[values],
                positions=[pos],
                widths=violin_width,
                showmeans=False,
                showmedians=False,
                showextrema=False
            )
            for pc in parts['bodies']:
                pc.set_facecolor(palette[group])
                pc.set_alpha(0.4)
                vertices = pc.get_paths()[0].vertices 
                vertices[:,0] = np.where( 
                    vertices[:,0] > violin_pos, 
                    vertices[:,0], 
                    violin_pos )


            # --- Box ---
            box = ax.boxplot(
                x=[values],
                positions=[pos],
                widths=box_width,
                patch_artist=True,
                showcaps=True,
                showfliers=False,
                notch = True
            )
            for patch in box['boxes']:
                patch.set_facecolor(palette[group])
                patch.set_alpha(0.8)
                patch.set_edgecolor(palette[group])
            for median in box['medians']:
                median.set_color('black')
                median.set_linewidth(3)

            # --- Scatter ---
            jitter = np.random.uniform(-0.03, 0.03, size=len(values))
            ax.scatter(
                np.full(len(values), pos-0.1) + jitter,
                values,
                color=palette[group],
                alpha=0.6,
                s=10,
                zorder=3
            )

    # --- Formatting ---
    ax.set_xticks(range(len(directions)))
    ax.set_xticklabels(directions)
    ax.set_title(f'{category} - Raincloud Plot')
    # ax.set_ylabel(category.split("_")[0].capitalize())
    # ax.set_xlabel('Direction')

    # Custom legend
    for i, group in enumerate(group_order):
        ax.scatter([], [], color=palette[group], label=group)
    ax.legend(title="Group", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(f'../res/09022025-pser-zstack/{category}_raincloud.png', dpi=300)
    plt.show()
