import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import average

def extract_vertical_midline_displacement(bos_file):
    df_BOS = pd.read_csv(bos_file, delimiter=',')

    x = df_BOS['x']
    y = df_BOS['y']
    u_final = df_BOS['x-displacement']
    v_final = df_BOS['y-displacement']

    # choose y row closest to 0
    x_mid = x.iloc[(x - 0).abs().argmin()]
    midline_mask = np.isclose(x, x_mid)

    y_mid = y[midline_mask]
    u_mid = u_final[midline_mask]
    v_mid = v_final[midline_mask]

    sort_idx = np.argsort(y_mid)
    y_mid = y_mid.values[sort_idx]
    u_mid = u_mid.values[sort_idx]
    v_mid = v_mid.values[sort_idx]

    midline_df = pd.DataFrame({
        'y': y_mid,
        'x_displacement': u_mid,
        'y_displacement': v_mid
    })

    return midline_df

def extract_horizontal_midline_displacement(bos_file):
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

def average_vector_ratio(x, y):
    # Mask out where there is absolutely no movement
    mask = (x != 0) | (y != 0)

    # Calculate angles only for moving pixels
    angles_radians = np.arctan(y[mask]/x[mask])
    angles_degrees = np.degrees(angles_radians)
    avg_deg = average(np.abs(angles_degrees))

    return avg_deg


image_no = 1
temp = 220


def sensitivity_graph_blur(a, b):
    avg_vert = []
    avg_horz = []

    for blur in b:
        bos_file = f"OF_dataframes (pixel method)/BOS_12_11_{image_no} ({temp}C) df with alpha {a[0]}, gaussian blur {blur}.csv"

        vertical_midline_df = extract_vertical_midline_displacement(bos_file)
        u_vert = vertical_midline_df['x_displacement']
        v_vert = vertical_midline_df['y_displacement']

        avg_ratio_vert = average_vector_ratio(u_vert, v_vert)
        avg_vert.append(avg_ratio_vert)

        horizontal_midline_df = extract_horizontal_midline_displacement(bos_file)
        u_horz = horizontal_midline_df['x_displacement']
        v_horz = horizontal_midline_df['y_displacement']

        print()

        avg_ratio_horz = average_vector_ratio(u_horz, v_horz)
        avg_horz.append(avg_ratio_horz)

    plt.plot(b, avg_vert, 'o', color = "red", label="Vertical midline")
    plt.plot(b, avg_horz, 'o', color = "blue", label="Horizontal midline")
    plt.xlabel("Gaussian blur parameter b")
    plt.ylabel("Average angle")
    plt.legend()
    plt.title(f"Blur parameter against the average angle of displacement vectors for alpha $\\alpha$ = {a[0]}")
    plt.show()


def sensitivity_graph_alpha(a, b):
    avg_vert = []
    avg_horz = []

    for alpha in a:
        bos_file = f"OF_dataframes (pixel method)/BOS_12_11_{image_no} ({temp}C) df with alpha {alpha}, gaussian blur {b[0]}.csv"

        vertical_midline_df = extract_vertical_midline_displacement(bos_file)
        u_vert = vertical_midline_df['x_displacement']
        v_vert = vertical_midline_df['y_displacement']

        avg_ratio_vert = average_vector_ratio(u_vert, v_vert)
        avg_vert.append(avg_ratio_vert)

        horizontal_midline_df = extract_horizontal_midline_displacement(bos_file)
        u_horz = horizontal_midline_df['x_displacement']
        v_horz = horizontal_midline_df['y_displacement']

        avg_ratio_horz = average_vector_ratio(u_horz, v_horz)
        avg_horz.append(avg_ratio_horz)

    plt.plot(a, avg_vert, 'o', color = 'red', label="Vertical midline")
    plt.plot(a, avg_horz, 'o', color = 'blue', label="Horizontal midline")
    plt.xlabel("Optical flow parameter alpha $\\alpha$")
    plt.ylabel("Average angle")
    plt.legend()
    plt.title(f"Alpha parameter against the average angle of displacement vectors for blur b = {b[0]}")
    plt.show()


a = [20]
b = [1, 3, 5, 7, 9, 11, 13, 15]

sensitivity_graph_blur(a, b)

b = [11]
a = [5, 10, 15, 20, 25, 30, 35, 40]

sensitivity_graph_alpha(a, b)
