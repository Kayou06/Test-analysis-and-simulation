import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from d02_Compressibility import compressibility_factor_and_density
from d02_field_corrections import mask_correction_densitygrad
from D02_data_postprocessing import get_images

Z, rho0 = compressibility_factor_and_density()


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


import numpy as np
import pandas as pd


def calc_drho_d(displacement, rho):
    C = 5.3
    ZD = 0.010
    ZA = 1.250
    f = 0.200
    W = 0.020
    K = 4.5e-4
    n0 = K * rho + 1
    return displacement * n0 * (ZD + ZA - f) / (C * W * K * f * ZD)


# def extract_all_displacements(bos_file, corr_file):
#     df_BOS = pd.read_csv(bos_file, delimiter=';')
#     df_corr = pd.read_csv(corr_file, delimiter=';')

#     x = df_BOS['x'].to_numpy()
#     y = df_BOS['y'].to_numpy()
#     u = df_BOS['x-displacement'].to_numpy()
#     v = df_BOS['y-displacement'].to_numpy()

#     u_corr = df_corr['x-displacement'].mean()
#     v_corr = df_corr['y-displacement'].mean()

#     u_final = u - u_corr
#     v_final = v - v_corr

#     return x, y, u_final, v_final


def calc_density_gradient_all_points(image_no, temp, alpha, blur, blur_type, rho, midpoint_finder=0):
    # midpoint_finder is either 0 (empty pixel method) or 1 (circle method)
    if midpoint_finder == 1:
        method = "circle method"
    else:
        method = "pixel method"

    file = f"OF_dataframes ({method})/BOS_12_11_{image_no} ({temp}C) df with alpha {alpha}, {blur_type} blur {blur}.csv"

    df_OF = pd.read_csv(file, delimiter=",")

    x = df_OF['x'].to_numpy()
    y = df_OF['y'].to_numpy()
    u_final = df_OF['x-displacement'].to_numpy()
    v_final = df_OF['y-displacement'].to_numpy()

    drho_dx = calc_drho_d(u_final, rho)
    drho_dy = calc_drho_d(v_final, rho)

    density_gradient = np.sqrt(drho_dx**2 + drho_dy**2)

    return x, y, density_gradient


def extract_midline_displacement(bos_file):
    df_BOS = pd.read_csv(bos_file, delimiter=',')

    x = df_BOS['x']
    y = df_BOS['y']
    u_final = df_BOS['x-displacement']
    v_final = df_BOS['y-displacement']

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


# # -------------------------------------------------
# # Plot drho/dx for all 7 files
# # -------------------------------------------------
# plt.figure(figsize=(10, 6))

# for bos_file, corr_file, rho in zip(bos_files, corr_files, rho0):
#     midline_df = extract_midline_displacement(bos_file, corr_file)

#     drho_dx = calc_drho_d(midline_df['x_displacement'], rho)
#     plt.plot(midline_df['x'], drho_dx, label=f'{bos_file.split("/")[-1]}')

# plt.xlabel('x')
# plt.ylabel('drho/dx')
# plt.title('drho/dx vs x for 7 BOS files')
# plt.grid(True)
# plt.legend()
# plt.show()


# -------------------------------------------------
# Plot normalized drho/dx for a single image
# -------------------------------------------------

def plot_normalized_drodx_OF(bos_file):
    plt.figure(figsize=(10, 6))

    midline_df = extract_midline_displacement(bos_file)

    drho_dx = calc_drho_d(midline_df['x_displacement'], rho)

    closest_idx = np.abs(midline_df['x']).argmin()
    drho_dx_at_x0 = drho_dx.iloc[closest_idx]

    normalized_drho_dx = -1*drho_dx / (rho*10**3)

    plt.plot(
        midline_df['x'],
        normalized_drho_dx,
        label=rf'$\rho_0 = {rho:.4f}\ \mathrm{{kg/m^3}}$'
    )

    plt.xlabel('x')
    plt.ylabel(r'Normalized $\frac{d\rho}{dx}$')
    plt.title(r'$\frac{d\rho}{dx}$ vs x at y = 0')

    plt.grid(True)
    plt.legend()
    plt.show()
 
image_no = 1
temp = 220
alpha = 35
blur = 11
blur_type = "gaussian"
rho = rho0[image_no - 1]
midpoint_finder = 0
# midpoint_finder is either 0 (empty pixel method) or 1 (circle method)
if midpoint_finder == 1:
    method = "circle method"
else:
    method = "pixel method"

bos_file = f"OF_dataframes ({method})/BOS_12_11_{image_no} ({temp}C) df with alpha {alpha}, {blur_type} blur {blur}.csv"

plot_normalized_drodx_OF(bos_file=bos_file)


# -------------------------------------------------
# Plot full density gradient field for a single image
# -------------------------------------------------

work_image_final, ref_image_final, temp = get_images(image_no)

x, y, density_gradient = calc_density_gradient_all_points(
    image_no=image_no, temp=temp, alpha=alpha, blur=blur, blur_type=blur_type, rho=rho)

density_gradient = mask_correction_densitygrad(density_gradient, ref_image_final)

# Keep only upper half
mask_upper = y >= 0

x_plot = x[mask_upper]
y_plot = y[mask_upper]
density_plot = density_gradient[mask_upper]

fig, ax = plt.subplots(figsize=(16, 4))

density_gradient_masked = np.ma.masked_where(
    density_plot ==0.,
    density_plot
)

cmap = plt.cm.viridis.copy()
cmap.set_bad(color="white")

sc = ax.scatter(x_plot, y_plot, c=density_gradient_masked, s=10, cmap=cmap)

ax.set_xlabel(r"$x$ [$mm$]")
ax.set_ylabel(r"$y$ [$mm$]")
ax.set_title("Density Gradient")
ax.set_aspect('equal', adjustable='box')

cbar = fig.colorbar(sc, ax=ax, orientation='horizontal', pad=0.25, fraction=0.1, aspect=60)
cbar.set_label(r"[$kg/m^3/mm$]")

# fig.savefig("FINAL PLOTS/Density Gradients/CC-densitygrad_BOS_12_11_1.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)

