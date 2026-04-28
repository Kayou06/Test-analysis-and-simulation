import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

rho0 = [
    40.27772187279685,
    75.04027767585774,
    139.0060691158923,
    209.16092353098253,
    91.60892909471781,
    59.861281984102675,
    41.428026313530424
]

def calc_drho_dx(del_x, rho):
    C = 5.3
    ZD = 0.010
    ZA = 1.250
    f = 0.200
    W = 0.020
    K = 4.5 * 10**-4
    n0 = K * rho + 1
    return del_x * n0 * (ZD + ZA - f) / (C * W * K * f * ZD)

all_x = []
all_y_raw = []
all_y_normalized = []

for i in range(1, 8):
    df_BOS = pd.read_csv(
        f'Raw_Pictures_Wavelet/BOS_12_11_{i}/BOS_12_11_{i}0001.csv',
        delimiter=';'
    )
    df_corr = pd.read_csv(
        f'Raw_Pictures_Wavelet/BOS_12_11_{i}/cross_correction0001.csv',
        delimiter=';'
    )

    x = df_BOS['x']
    y = df_BOS['y']
    u = df_BOS['x-displacement']
    v = df_BOS['y-displacement']

    u_corr = df_corr['x-displacement'].mean()
    v_corr = df_corr['y-displacement'].mean()

    u_final = u - u_corr
    v_final = v - v_corr

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

    midline_df['drho_dx'] = calc_drho_dx(midline_df['x_displacement'], rho0[i - 1])

    closest_idx = (midline_df['x'] - 0).abs().argmin()
    drho_dx_at_x0 = midline_df.loc[closest_idx, 'drho_dx']

    midline_df['normalized_drho_dx'] = midline_df['drho_dx'] / drho_dx_at_x0

    all_x.append(midline_df['x'].to_numpy())
    all_y_raw.append(midline_df['drho_dx'].to_numpy())
    all_y_normalized.append(midline_df['normalized_drho_dx'].to_numpy())

x_common = all_x[0]

all_curves_raw = np.array([
    np.interp(x_common, all_x[i], all_y_raw[i]) for i in range(len(all_y_raw))
])

all_curves_normalized = np.array([
    np.interp(x_common, all_x[i], all_y_normalized[i]) for i in range(len(all_y_normalized))
])

# Plot non-normalized curves
plt.figure()
for i in range(len(all_curves_raw)):
    plt.plot(x_common, all_curves_raw[i], label=f'rho0={rho0[i]}')

plt.xlabel(r'x')
plt.ylabel(r'$\frac{d\rho}{dx}$')
plt.title(r'$\frac{d\rho}{dx}$ vs x')
plt.legend()
plt.grid(True)
plt.savefig(f"FINAL PLOTS/Midline Density Gradient/CC-densitygrad_y0.png", dpi=300)
plt.show()

# Plot normalized curves
plt.figure()
for i in range(len(all_curves_normalized)):
    plt.plot(x_common, all_curves_normalized[i], label=f'rho0={rho0[i]}')

plt.xlabel(r'x')
plt.ylabel(r'Normalized $\frac{d\rho}{dx}$')
plt.title(r'Normalized $\frac{d\rho}{dx}$ vs x')
plt.legend()
plt.grid(True)
plt.savefig(f"FINAL PLOTS/Midline Density Gradient/CC-Normalized_densitygrad_y0.png", dpi=300)
plt.show()

# Max distance between normalized curves
distance_normalized = all_curves_normalized.max(axis=0) - all_curves_normalized.min(axis=0)
max_distance_normalized = distance_normalized.max()

print("Max distance between normalized curves:", max_distance_normalized)

# Max distance between non-normalized curves
distance_raw = all_curves_raw.max(axis=0) - all_curves_raw.min(axis=0)


# Plot non-normalized curves with upper/lower uncertainty bounds
plt.figure()

# Plot each non-normalized curve with its uncertainty separately
for i in range(1, 8):
    curve_idx = i - 1

    y_curve = all_curves_raw[curve_idx]

    # Convert normalized distance to raw scale for this rho0
    uncertainty = distance_normalized * rho0[curve_idx]

    upper = y_curve + uncertainty
    lower = y_curve - uncertainty

    plt.figure()

    plt.plot(
        x_common,
        y_curve,
        label=f'rho0={rho0[curve_idx]}'
    )

    plt.plot(
        x_common,
        upper,
        '--',
        label='Upper bound'
    )

    plt.plot(
        x_common,
        lower,
        '--',
        label='Lower bound'
    )

    plt.xlabel(r'x')
    plt.ylabel(r'$\frac{d\rho}{dx}$')
    plt.title(rf'$\frac{{d\rho}}{{dx}}$ with uncertainty (rho0={rho0[curve_idx]:.2f})')
    plt.legend()
    plt.grid(True)

    plt.savefig(
        f"FINAL PLOTS/Midline Density Gradient/CC-densitygrad_uncertainty_y0_case_{i}.png",
        dpi=300
    )

    plt.show()