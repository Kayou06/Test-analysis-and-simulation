import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

def blur_func(data_raw):
    data_smooth = savgol_filter(data_raw, window_length=71, polyorder=3)
    return data_smooth


def plot_OF(alpha, blur, blur_type, type=0):
    # midpoint_finder is either 0 (empty pixel method) or 1 (circle method)
    if type == 1:
        method = "circle method"
    else:
        method = "pixel method"

    # rho0 = [40.27772187279685, 75.04027767585774, 139.0060691158923, 209.16092353098253, 91.60892909471781, 59.861281984102675, 41.428026313530424]
    rho0 = np.ones(7)

    def calc_drho_dx(del_x, rho):
        C = 5.3
        ZD = 0.010
        ZA = 1.250
        f = 0.200
        W = 0.020
        K = 4.5*10**-4
        n0 = K*rho+1
        return del_x * n0 * (ZD + ZA - f) / (C * W * K * f * ZD)

    all_x = []
    all_y = []

    for i in range(1, 8):
    #i = 1
    #while i == 1:

        if i == 1 or i == 2:
            temp = 220
        else:
            temp = 252

        file = f"OF_dataframes ({method})/BOS_12_11_{i} ({temp}C) df with alpha {alpha}, {blur_type} blur {blur}.csv"
        #file = f"Midline_displacements (pixel method)\Image_{i}.csv"

        # get coordinates and displacements, coverting from mm to m
        x = df_OF['x'] * 10**(-3)
        y = df_OF['y'] * 10**(-3)
        u_final = df_OF['x-displacement'] * 10**(-3)
        v_final = df_OF['y-displacement'] * 10**(-3)

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

        drho_dx = calc_drho_dx(midline_df['x_displacement'], rho0[i-1])
        drho_dx = blur_func(drho_dx)

        midline_df['drho_dx'] = drho_dx 

        closest_idx = (midline_df['x'] - 0).abs().argmin()
        drho_dx_at_x0 = midline_df.loc[closest_idx, 'drho_dx']

        midline_df['normalized_drho_dx'] = (midline_df['drho_dx'] / rho0[i-1])

        all_x.append(midline_df['x'].to_numpy())
        all_y.append(midline_df['normalized_drho_dx'].to_numpy())

        i+=1

    x_common = all_x[0]

    all_curves = np.array([
        np.interp(x_common, all_x[i], all_y[i]) for i in range(len(all_y))
    ])

    for i in range(len(all_curves)):
        plt.plot(x_common, all_curves[i], label=f'rho0={rho0[i]}')


    distance = all_curves.max(axis=0) - all_curves.min(axis=0)
    max_distance = distance.max()


    print("Max distance between normalized curves:", max_distance)
    return all_curves

plot_OF(alpha=35, blur=11, blur_type="gaussian")

plt.xlabel('x')
plt.ylabel(r'Normalized $\frac{d\rho}{dx}$')
plt.title(r'Normalized $\frac{d\rho}{dx}$ vs x')
plt.grid(True)
plt.show()