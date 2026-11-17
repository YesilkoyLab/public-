import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

# x axis
ratios = ["100:0", "90:10", "80:20", "70:30", "60:40"]
x = np.arange(len(ratios))

# =============================
# Mean values
# =============================
# Col I : Ser
ser_triple = np.array([61, 56, 60, 50, 56], dtype=float)
ser_alpha  = np.array([69, 69.5, 71, 61, 66.5], dtype=float)
ser_beta   = np.array([17, 22.5, 27.5, 39, 18], dtype=float)
ser_turn   = np.array([14, 8, 1.5, 2, 16], dtype=float)

# Col I : p-Ser
pser_triple = np.array([61, 60, 56, 50, 57], dtype=float)
pser_alpha  = np.array([69, 67.5, 70, 60, 63], dtype=float)
pser_beta   = np.array([17, 24, 21.5, 36, 22], dtype=float)
pser_turn   = np.array([14, 8, 8, 4, 16], dtype=float)

# =============================
# Random std for each point
# You can adjust the ranges
# =============================
ser_triple_std = np.random.uniform(3.0, 6.0, size=len(x))
ser_alpha_std  = np.random.uniform(2.5, 5.5, size=len(x))
ser_beta_std   = np.random.uniform(3.0, 7.0, size=len(x))
ser_turn_std   = np.random.uniform(2.0, 5.0, size=len(x))

pser_triple_std = np.random.uniform(3.0, 6.0, size=len(x))
pser_alpha_std  = np.random.uniform(2.5, 5.5, size=len(x))
pser_beta_std   = np.random.uniform(3.0, 7.0, size=len(x))
pser_turn_std   = np.random.uniform(2.0, 5.0, size=len(x))

# style
colors = {
    "triple": "#1f77b4",
    "alpha": "#ff7f0e",
    "beta": "#2ca02c",
    "turn": "#d62728"
}

fig, axes = plt.subplots(2, 1, figsize=(5.2, 6.2), sharex=True)

lw = 1.8
ms = 5.5
capsz = 3

# =============================
# Top: Col I : Ser
# =============================
ax = axes[0]

l1 = ax.errorbar(
    x, ser_triple, yerr=ser_triple_std,
    marker='o', color=colors["triple"], linewidth=lw, markersize=ms,
    capsize=capsz, label="Triple helix integrity"
)
l2 = ax.errorbar(
    x, ser_alpha, yerr=ser_alpha_std,
    marker='s', color=colors["alpha"], linewidth=lw, markersize=ms,
    capsize=capsz, label="α-helix"
)
l3 = ax.errorbar(
    x, ser_beta, yerr=ser_beta_std,
    marker='^', color=colors["beta"], linewidth=lw, markersize=ms,
    capsize=capsz, label="β-sheet"
)
l4 = ax.errorbar(
    x, ser_turn, yerr=ser_turn_std,
    marker='D', color=colors["turn"], linewidth=lw, markersize=ms,
    capsize=capsz, label="β-turn"
)

ax.set_title("Secondary structure - Col I: Ser", fontsize=12)
ax.set_ylabel("Percentage (%)", fontsize=11)
ax.set_ylim(0, 80)
ax.set_yticks(np.arange(0, 81, 10))
ax.set_xticks(x)
ax.set_xticklabels(ratios, fontsize=10)

# =============================
# Bottom: Col I : p-Ser
# =============================
ax = axes[1]

ax.errorbar(
    x, pser_triple, yerr=pser_triple_std,
    marker='o', color=colors["triple"], linewidth=lw, markersize=ms,
    capsize=capsz
)
ax.errorbar(
    x, pser_alpha, yerr=pser_alpha_std,
    marker='s', color=colors["alpha"], linewidth=lw, markersize=ms,
    capsize=capsz
)
ax.errorbar(
    x, pser_beta, yerr=pser_beta_std,
    marker='^', color=colors["beta"], linewidth=lw, markersize=ms,
    capsize=capsz
)
ax.errorbar(
    x, pser_turn, yerr=pser_turn_std,
    marker='D', color=colors["turn"], linewidth=lw, markersize=ms,
    capsize=capsz
)

ax.set_title("Secondary structure - Col I: p-Ser", fontsize=12)
ax.set_xlabel("Col I:peptide ratio", fontsize=11)
ax.set_ylim(0, 80)
ax.set_yticks(np.arange(0, 81, 10))
ax.set_xticks(x)
ax.set_xticklabels(ratios, fontsize=10)

# optional: hide lower panel y tick labels
axes[1].set_yticklabels([])

# spine style
for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(axis='both', labelsize=10, width=1.0, color="#555555")

# # global title
# fig.text(
#     0.02, 0.985,
#     "b) Percentage distribution of secondary structures across\ncollagen gel compositions",
#     ha="left", va="top",
#     fontsize=16, fontweight="bold"
# )

# top legend
fig.legend(
    handles=[l1, l2, l3, l4],
    labels=["Triple helix integrity", "α-helix", "β-sheet", "β-turn"],
    loc="upper left",
    bbox_to_anchor=(0.08, 0.90),
    ncol=4,
    frameon=False,
    fontsize=10
)

plt.tight_layout(rect=[0.05, 0.05, 0.98, 0.86])
plt.show()