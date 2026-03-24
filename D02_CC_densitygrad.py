import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


rho0 = [
    40.27772187279685,
    75.04027767585774,
    139.0060691158923,
    209.16092353098253,
    91.60892909471781,
    59.861281984102675,
    41.428026313530424
]

bos_files = [
    'Raw_Pictures_Wavelet/BOS_12_11_1/BOS_12_11_10001.csv',
    'Raw_Pictures_Wavelet/BOS_12_11_2/BOS_12_11_20001.csv',
    'Raw_Pictures_Wavelet/BOS_12_11_3/BOS_12_11_30001.csv',
    'Raw_Pictures_Wavelet/BOS_12_11_4/BOS_12_11_40001.csv',
    'Raw_Pictures_Wavelet/BOS_12_11_5/BOS_12_11_50001.csv',
    'Raw_Pictures_Wavelet/BOS_12_11_6/BOS_12_11_60001.csv',
    'Raw_Pictures_Wavelet/BOS_12_11_7/BOS_12_11_70001.csv'
]

corr_files = [
    'Raw_Pictures_Wavelet/BOS_12_11_1/cross_correction0001.csv',
    'Raw_Pictures_Wavelet/BOS_12_11_2/cross_correction0001.csv',
    'Raw_Pictures_Wavelet/BOS_12_11_3/cross_correction0001.csv',
    'Raw_Pictures_Wavelet/BOS_12_11_4/cross_correction0001.csv',
    'Raw_Pictures_Wavelet/BOS_12_11_5/cross_correction0001.csv',
    'Raw_Pictures_Wavelet/BOS_12_11_6/cross_correction0001.csv',
    'Raw_Pictures_Wavelet/BOS_12_11_7/cross_correction0001.csv'
]


def calc_drho_dx(del_x, rho):
    C = 5.3
    ZD = 0.010
    ZA = 1.250
    f = 0.200
    W = 0.020
    K = 4.5e-4
    n0 = K * rho + 1
    return del_x * n0 * (ZD + ZA - f) / (C * W * K * f * ZD)


def extract_midline_displacement(bos_file, corr_file):
    df_BOS = pd.read_csv(bos_file, delimiter=';')
    df_corr = pd.read_csv(corr_file, delimiter=';')

    x = df_BOS['x']
    y = df_BOS['y']
    u = df_BOS['x-displacement']
    v = df_BOS['y-displacement']

    u_corr = df_corr['x-displacement'].mean()
    v_corr = df_corr['y-displacement'].mean()

    u_final = u - u_corr
    v_final = v - v_corr

    # choose y row closest to 0
    y_mid = y.iloc[(y - 0).abs().argmin()]
    midline_mask = np.isclose(y, y_mid)

    x_mid = x[midline_mask]
    u_mid = u_final[midline_mask]
    v_mid = v_final[midline_mask]

    sort_idx = np.argsort(x_mid)
    x_mid = x_mid.values[sort_idx]
    u_mid = u_mid.values[sort_idx]
    v_mid = v_mid.values[sort_idx]

    midline_df = pd.DataFrame({
        'x': x_mid,
        'x_displacement': u_mid,
        'y_displacement': v_mid
    })

    return midline_df


# -------------------------------------------------
# Plot drho/dx for all 7 files
# -------------------------------------------------
plt.figure(figsize=(10, 6))

for bos_file, corr_file, rho in zip(bos_files, corr_files, rho0):
    midline_df = extract_midline_displacement(bos_file, corr_file)

    drho_dx = calc_drho_dx(midline_df['x_displacement'], rho)
    plt.plot(midline_df['x'], drho_dx, label=f'{bos_file.split("/")[-1]}')

plt.xlabel('x')
plt.ylabel('drho/dx')
plt.title('drho/dx vs x for 7 BOS files')
plt.grid(True)
plt.legend()
plt.show()


# -------------------------------------------------
# Plot normalized drho/dx for all 7 files
# -------------------------------------------------
plt.figure(figsize=(10, 6))

for bos_file, corr_file, rho in zip(bos_files, corr_files, rho0):
    midline_df = extract_midline_displacement(bos_file, corr_file)

    drho_dx = calc_drho_dx(midline_df['x_displacement'], rho)

    closest_idx = np.abs(midline_df['x']).argmin()
    drho_dx_at_x0 = drho_dx.iloc[closest_idx]

    normalized_drho_dx = drho_dx / drho_dx_at_x0

    plt.plot(midline_df['x'], normalized_drho_dx, label=f'{bos_file.split("/")[-1]}')

plt.xlabel('x')
plt.ylabel('Normalized drho/dx')
plt.title('Normalized drho/dx vs x for 7 BOS files')
plt.grid(True)
plt.legend()
plt.show()