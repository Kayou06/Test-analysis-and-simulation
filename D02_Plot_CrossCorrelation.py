import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
def plot_CC():
    rho0 = [40.27772187279685, 75.04027767585774, 139.0060691158923, 209.16092353098253, 91.60892909471781, 59.861281984102675, 41.428026313530424]

    def calc_drho_dx(del_x, rho):
        C = 5.3
        ZD = 0.010
        ZA = 1.250
        f = 0.200
        W = 0.020
        K = 4.5*10**-4
        n0 = K*rho+1
        return del_x * n0 * (ZD + ZA - f) / (C * W * K * f * ZD)/1000

    all_x = []
    all_y = []
    all_ex = []
    all_x_OF = []
    all_y_OF = []


    for i in range(1, 8):
        df_BOS = pd.read_csv(f'Raw_Pictures_Wavelet/BOS_12_11_{i}/BOS_12_11_{i}0001.csv', delimiter=';')
        df_corr = pd.read_csv(f'Raw_Pictures_Wavelet/BOS_12_11_{i}/cross_correction0001.csv', delimiter=';')
        of_df= pd.read_csv(f'D02_OF_CURVES/Image_{i}.csv', delimiter = ',')
        x = df_BOS['x']
        y = df_BOS['y']
        u = df_BOS['x-displacement']
        v = df_BOS['y-displacement']
        ex = df_BOS['Uncertainty Vx']
        ey = df_BOS['Uncertainty Vy']

        u_corr = df_corr['x-displacement'].mean()
        v_corr = df_corr['y-displacement'].mean()

        u_final = u - u_corr
        v_final = v - v_corr

        y_mid = y.iloc[(y - 0).abs().argmin()]
        midline_mask = np.isclose(y, y_mid)

        x_mid = x[midline_mask]
        u_mid = u_final[midline_mask]
        v_mid = v_final[midline_mask]
        ex_mid = ex[midline_mask]
        ey_mid = ey[midline_mask]

        sort_idx = np.argsort(x_mid)
        x_mid = x_mid.values[sort_idx]
        u_mid = u_mid.values[sort_idx]
        v_mid = v_mid.values[sort_idx]
        ex_mid = ex_mid.values[sort_idx]
        ey_mid = ey_mid.values[sort_idx]

        midline_df = pd.DataFrame({
            'x': x_mid,
            'x_displacement': u_mid,
            'y_displacement': v_mid,
            'Uncertainty Vx': ex_mid,
            'Uncertainty Vy': ey_mid
        })

        

        drho_dx    = calc_drho_dx(u_mid,  rho0[i - 1]) / rho0[i - 1]
        drho_dx_OF    = calc_drho_dx(of_df['ux'],  rho0[i - 1]) / rho0[i - 1]
        drho_dx_ex = calc_drho_dx(ex_mid, rho0[i - 1]) / rho0[i - 1]

        all_x.append(midline_df['x'].to_numpy())
        all_x_OF.append(of_df['x'].to_numpy())
        all_y.append(drho_dx)
        all_y_OF.append(drho_dx_OF)
        all_ex.append(drho_dx_ex)


    x_common = all_x[0]
    x_common_OF = all_x_OF[0]
    all_curves_OF = np.array([
        np.interp(x_common, all_x_OF[i], all_y_OF[i]) for i in range(len(all_y))
    ])
    all_curves = np.array([
        np.interp(x_common, all_x[i], all_y[i]) for i in range(len(all_y))
    ])
    all_ex_interp = np.array([np.interp(x_common, all_x[i], all_ex[i]) for i in range(len(all_ex))])
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes = axes.flatten()

    for i, ax in enumerate(axes[:7]):       # first 7 panels
        curve = all_curves[i]
        curve_OF = all_curves_OF[i]
        err   = all_ex_interp[i]

        ax.plot(x_common, curve, label=f'rho0={rho0[i]:.2f}')
        ax.plot(x_common_OF, curve_OF, label=f'rho0={rho0[i]:.2f}')
        ax.fill_between(x_common, curve - err, curve + err,
                        alpha=0.3, label='Uncertainty')
        ax.set_xlabel('x')
        ax.set_ylabel(r'Normalized $\frac{d\rho}{dx}$')
        ax.set_title(rf'$\rho_0$={rho0[i]:.2f}')
        ax.grid(True)
        ax.legend()

    axes[7].set_visible(False)              # hide the empty 8th panel

    plt.tight_layout()
    plt.show()


    return all_curves, all_ex_interp
plot_CC()

