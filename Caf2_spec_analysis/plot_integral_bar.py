import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# liver ffpe
# y = np.array([5.463530189, 1.999999958, 8.692317924, 11.41583379, 3.284611429, 24.3547364, 0.979385829])
# std = np.array([0.248548151, 8.07E-08, 0.372347184, 0.775681519, 0.991578269, 1.977680634, 0.435252128])

# # kidney ffpe
# y = np.array([1.701224527,
# 5.874700573,
# 6.60525563,
# 2.628001717,
# 13.50298168,
# 1.054720217])
# std = np.array([0.083527576,
# 0.277252345,
# 0.466000032,
# 0.130855699,
# 0.982329792,
# 0.075842295])

# # kidney oct
# x = np.array([1080,
# 1240,
# 1404,
# 1462,
# 1546,
# 1584,
# 1656,
# 1744])
# y=np.array([6.079406529,
# 7.466440577,
# 5.680016109,
# 5.783433695,
# 13.05460871,
# 4.877807878,
# 20.64610274,
# 1.615591127])
# std = np.array([2.427444716,
# 2.291782154,
# 1.536940682,
# 1.561840744,
# 4.356907829,
# 1.473909472,
# 6.897478695,
# 0.314954989])

# # liver oct
# x=np.array([1029.999977,
# 1080.000002,
# 1152.000038,
# 1239.999937,
# 1308.000009,
# 1345.999991,
# 1400,
# 1464,
# 1546.000224,
# 1584.000001,
# 1656,
# 1744.021709])
# y=np.array([7.861133591,
# 7.63821075,
# 7.907711492,
# 10.89160374,
# 7.152571329,
# 7.177070279,
# 4.875001028,
# 6.787829514,
# 19.13975724,
# 7.246205763,
# 41.7555259,
# 2.973826497])
# std = np.array([0.323639023,
# 0.176188769,
# 0.214959236,
# 0.271280514,
# 0.181171244,
# 0.174400235,
# 0.104888012,
# 1.802141087,
# 3.75285113,
# 0.144825654,
# 1.178783241,
# 0.024883852])

# --- Load your CSV file ---
foldername_list = ['HMT_1/','HMT_2/','HMT_3/','HMT_4/','HMT_5/','HMT_6/']
folder_path = 'D:/res/rat_integral/liver_ff/'
for file in foldername_list:
    path = folder_path + file
    df = pd.read_csv(path + 'summary_statistics.csv')  # Change to your file name if different/ summary_statistics.csv integral_value_summary.csv
    # Your data
    x = np.array([968, 1030, 1082, 1154, 1238, 1308, 1400, 1448, 1464, 1516, 1546, 1656, 1744]) #968, 1036, 1082, 1168, 1238, 1310, 1400,
    bar_width = 25
    # --- Extract y (mean) and std from CSV ---
    # y = df['mean'].to_numpy()
    # std = df['std'].to_numpy()
    y = df['Average Integral'].to_numpy()
    std = df['Standard Deviation'].to_numpy()


    # Bar plot
    plt.figure(figsize=(12, 3))
    plt.bar(x, y, width=bar_width, yerr=std, capsize=5, color='grey', edgecolor='black')

    # Labels and formatting
    plt.xlim(950, 1800)
    plt.ylim(0, 100)
    # plt.xlabel('Wavenumber (cm⁻¹)', fontsize=14)
    plt.ylabel('Integral', fontsize=24)
    # plt.title('Feature Importance at Selected Wavenumbers', fontsize=16)
    plt.xticks(x, fontsize=18,rotation=30)
    plt.yticks(fontsize=18)
    # plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(path, f"fitting_integral.png"))
    # plt.show()
