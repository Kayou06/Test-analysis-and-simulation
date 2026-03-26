import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from Compressibility import compressibility_factor_and_density
from D02_CC_densitygrad import calc_drho_dx

Z, rho0 = compressibility_factor_and_density()

def pixel_to_coords(mask_point, SF=25.097):

    x_coords = mask_point[:, 0]
    y_coords = mask_point[:, 1]

    heights = []
    x_valid = []

    bin_size = 1  # pixel resolution

    x_min = np.min(x_coords)
    x_max = np.max(x_coords)

    bins = np.arange(x_min, x_max + bin_size, bin_size)

    for i in range(len(bins) - 1):
        mask = (x_coords >= bins[i]) & (x_coords < bins[i+1])
        y_slice = y_coords[mask]

        if len(y_slice) < 2:
            continue

        y_min = np.min(y_slice)
        y_max = np.max(y_slice)

        height = y_max - y_min

        heights.append(height)
        x_valid.append(0.5 * (bins[i] + bins[i+1]))  # bin center

    heights = np.array(heights)
    x_valid = np.array(x_valid)

    throat_idx = np.argmin(heights)

    x_throat = x_valid[throat_idx]
    mask = np.abs(x_coords - x_throat) < bin_size
    y_throat = 0.5 * (np.max(y_coords[mask]) + np.min(y_coords[mask]))

    return x_throat, y_throat



def build_dataframe(u_corr, v_corr, image_no, SF=25.097):
    # SF = 25.097 [px/mm]

    if image_no == 1 or image_no == 2:
        mask_point = np.load(f"Mask_shapes/theBOSmask220C.npy")
    else:
        mask_point = np.load(f"Mask_shapes/Mask_252.npy")
    
    mask_len = np.size(mask_point)
    mask_point = mask_point.reshape(int(mask_len/2),2)


    x_throat, y_throat = pixel_to_coords(mask_point)
    
    rows, cols = u_corr.shape

    x_coords = (np.arange(cols) - x_throat) / SF
    y_coords = (np.arange(rows) - y_throat) / SF

    print(f"x range: {x_coords.min()} to {x_coords.max()} mm")
    print(f"y range: {y_coords.min()} to {y_coords.max()} mm")

    X, Y = np.meshgrid(x_coords, y_coords)


    OF_df = pd.DataFrame({
        'x': X.flatten(),  # x-coordinates
        'y': Y.flatten(),  # y-coordinates
        'x_displacement_corrected': u_corr.flatten(),  # x-displacement values
        'y_displacement_corrected': v_corr.flatten()  # y-displacement values
    })

    return OF_df

def test_funcs():
    image_no = 1
    temp = 220
    alpha = 35
    blur = 11
    blur_type = "gaussian"

    file3 = Path(f"VF BOS_12_11_{image_no} ({temp}) corrected/u_HS_alpha{alpha}_blur{blur}_{blur_type}.npy")
    file4 = Path(f"VF BOS_12_11_{image_no} ({temp}) corrected/v_HS_alpha{alpha}_blur{blur}_{blur_type}.npy")

    u_corr = np.load(file3)
    v_corr = np.load(file4)

    OF_df = build_dataframe(u_corr, v_corr, image_no)
    
    plt.figure()
    sc = plt.scatter(
        OF_df['x'],
        OF_df['y'],
        c=OF_df['x_displacement_corrected'],
        s=1
    )
    plt.colorbar(sc, label='x displacement')
    plt.xlabel('x [mm]')
    plt.ylabel('y [mm]')
    plt.title('Displacement field (centered at throat)')
    plt.gca().invert_yaxis()  # matches image coordinates
    plt.show()

test_funcs()

def extract_midline_displacement_OF():
    a=1



def plot_OF_densitygrad(image_no, alpha, blur, blur_type):
    rho = rho0[image_no - 1]

    midline_df = extract_midline_displacement_OF()
    drho_dx = calc_drho_dx(midline_df['x_displacement'], rho)

    closest_idx = np.abs(midline_df['x']).argmin()
    drho_dx_at_x0 = drho_dx.iloc[closest_idx]

    normalized_drho_dx = drho_dx / drho_dx_at_x0