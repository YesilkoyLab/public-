import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Data
# -----------------------------
ratios = ['100:0', '90:10', '80:20', '70:30', '60:40']
x = np.arange(len(ratios))

# Ser
ser_triple = [91, 86, 90, 80, 86]
ser_triple_err = [4, 6, 5, 4, 3]

ser_alpha = [69, 69.5, 71, 61, 66]
ser_alpha_err = [3, 3, 5, 4, 4]

ser_beta_sheet = [17, 23, 28, 39, 18]
ser_beta_sheet_err = [3, 6, 6, 4, 4]

ser_beta_turn = [14, 8, 2, 2, 16]
ser_beta_turn_err = [3, 3, 3, 4, 2]

# p-Ser
pser_triple = [90, 89, 85, 79, 86]
pser_triple_err = [4, 3, 3, 4, 4]

pser_alpha = [68, 67, 69, 58, 61]
pser_alpha_err = [4, 3, 4, 5, 3]

pser_beta_sheet = [16, 22, 20, 35, 20]
pser_beta_sheet_err = [4, 4, 3, 7, 6]

pser_beta_turn = [14, 6, 6, 3, 15]
pser_beta_turn_err = [4, 3, 2, 4, 4]

# -----------------------------
# Plot
# -----------------------------
fig, axes = plt.subplots(
    2, 1, figsize=(6.2, 7.0), sharex=True, sharey=True
)

# ----- Top: Ser -----
axes[0].errorbar(
    x, ser_triple, yerr=ser_triple_err, fmt='o-', capsize=5,
    linewidth=2, markersize=8, label='Triple helix integrity'
)
axes[0].errorbar(
    x, ser_alpha, yerr=ser_alpha_err, fmt='s-', capsize=5,
    linewidth=2, markersize=8, label='α-helix'
)
axes[0].errorbar(
    x, ser_beta_sheet, yerr=ser_beta_sheet_err, fmt='^-', capsize=5,
    linewidth=2, markersize=8, label='β-sheet'
)
axes[0].errorbar(
    x, ser_beta_turn, yerr=ser_beta_turn_err, fmt='D-', capsize=5,
    linewidth=2, markersize=8, label='β-turn'
)
axes[0].set_title('Secondary structure - Col I: Ser', fontsize=20)
axes[0].set_ylabel('Percentage (%)', fontsize=20)

# ----- Bottom: p-Ser -----
axes[1].errorbar(
    x, pser_triple, yerr=pser_triple_err, fmt='o-', capsize=5,
    linewidth=2, markersize=8, label='Triple helix integrity'
)
axes[1].errorbar(
    x, pser_alpha, yerr=pser_alpha_err, fmt='s-', capsize=5,
    linewidth=2, markersize=8, label='α-helix'
)
axes[1].errorbar(
    x, pser_beta_sheet, yerr=pser_beta_sheet_err, fmt='^-', capsize=5,
    linewidth=2, markersize=8, label='β-sheet'
)
axes[1].errorbar(
    x, pser_beta_turn, yerr=pser_beta_turn_err, fmt='D-', capsize=5,
    linewidth=2, markersize=8, label='β-turn'
)
axes[1].set_title('Secondary structure - Col I: p-Ser', fontsize=20)
axes[1].set_xlabel('Col I:peptide ratio', fontsize=20)

# -----------------------------
# Shared formatting
# -----------------------------
for ax in axes:
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_xticks(x)
    ax.set_xticklabels(ratios, fontsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Global legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc='upper center',
    ncol=4,
    frameon=False,
    bbox_to_anchor=(0.5, 0.98),
    fontsize=20
)

plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.show()

# Optional save
# fig.savefig('secondary_structure_ser_pser.png', dpi=300, bbox_inches='tight')