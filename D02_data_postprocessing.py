import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from Pre_processing import standard_pre
from Masking import shape_isolation
from Pyramidal_Horn_Schunck_tqdm import reshape
import pandas as pd
from d02_field_corrections import mask_correction
from d02_circle_midline_finder import find_midline

# This file contains functions which perform post-processing for OF data:
# finding the throat position, converting pixel coordinates to reas-world coordinates, and building a dataframe with the results

# get masked and cropped images, computed during the preprocessing step
def get_images(image_number):
    work_img_final = cv.imread(f"Correlable_pics/BOS_12_11_{image_number}_masked.tif")
    
    if image_number == 1 or image_number == 2:
        temp = 220
        ref_img_final = cv.imread(f"Correlable_pics/BOS_12_11_ref_masked ({temp}C).tif", cv.IMREAD_GRAYSCALE)
    else:
        temp = 252
        ref_img_final = cv.imread(f"Correlable_pics/BOS_12_11_ref_masked ({temp}C).tif", cv.IMREAD_GRAYSCALE)

    return work_img_final, ref_img_final, temp


# find throat position -> empty pixel method
def find_throat_position(work_img_cropped):
    empty_counts = np.sum(work_img_cropped == 255, axis=0)

    x_min = 200
    x_max = 2200
    restricted_counts = empty_counts[x_min:x_max]

    throat_x_relative = np.argmax(restricted_counts)

    throat_x = throat_x_relative + x_min
    
    column = work_img_cropped[:, throat_x]
    nozzle_pixels = np.where(column != 255)[0]

    if len(nozzle_pixels) == 0:
        throat_y = None
    else:
        y_top = nozzle_pixels[0]
        y_bottom = nozzle_pixels[-1]
        throat_y = int((y_top + y_bottom) / 2)

    return throat_x, throat_y

# convert pixel coordinates to regular coordinates (with standard scaling factor)
def pixel_to_coords(x_pixels, y_pixels, throat_x, throat_y, SF=25.097):
    # SF = 25.097 [px/mm]

    x_pixels_grid, y_pixels_grid = np.meshgrid(
                                   np.arange(x_pixels), 
                                   np.arange(y_pixels))

    x_coords = (x_pixels_grid - throat_x) / SF
    y_coords = (y_pixels_grid - throat_y) / SF

    return x_coords, y_coords

# image_no = 1
# alpha = 35
# blur = 15
# blur_type = "gaussian"

# work_img_final, ref_img_final, temp = get_images(image_no)
# u = np.load(f"VF BOS_12_11_{image_no} ({temp}) corrected/u_HS_alpha{alpha}_blur{blur}_{blur_type}.npy")
# v = np.load(f"VF BOS_12_11_{image_no} ({temp}) corrected/v_HS_alpha{alpha}_blur{blur}_{blur_type}.npy")

# # scaling factor [px/mm]
# SF = 25.097

# u_reshaped, v_reshaped = reshape(u, v, ref_img_final)
# u_reshaped = u_reshaped / SF
# v_reshaped = v_reshaped / SF
# throat_x, throat_y = find_throat_position(ref_img_final)
# y_pixels, x_pixels = u_reshaped.shape
# x_coords, y_coords = pixel_to_coords(x_pixels, y_pixels, throat_x, throat_y)


def build_dataframe(x, y, ux, uy):
    df = pd.DataFrame({
        "x": x.flatten(),
        "y": y.flatten(),
        "x-displacement": ux.flatten(),
        "y-displacement": uy.flatten()
    })
    return df


def data_postprocessing(image_no, u, v, type=0):
    path = f"Raw_Pictures_Wavelet/BOS_12_11_{image_no}.tif"

    work_img_final, ref_img_final, temp = get_images(image_no)

    # scaling factor [px/mm]
    SF = 25.097

    u_reshaped, v_reshaped = reshape(u, v, ref_img_final)
    u_reshaped, v_reshaped = mask_correction(u_reshaped, v_reshaped, ref_img_final)

    u_reshaped = u_reshaped / SF
    v_reshaped = v_reshaped / SF

    if type ==0:
        throat_x, throat_y = find_throat_position(ref_img_final)
    elif type == 1:
        throat_x, throat_y = find_midline(path)

    y_pixels, x_pixels = u_reshaped.shape
    x_coords, y_coords = pixel_to_coords(x_pixels, y_pixels, throat_x, throat_y)

    return x_coords, y_coords, u_reshaped, v_reshaped