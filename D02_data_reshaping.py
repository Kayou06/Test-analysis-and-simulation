import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from Pre_processing import standard_pre
from Masking import shape_isolation
from Pyramidal_Horn_Schunck_tqdm import reshape
import pandas as pd


def get_images(image_number):
    work_img_final = cv.imread(f"Correlable_pics/BOS_12_11_{image_number}_masked.tif")
    
    if image_number == 1 or image_number == 2:
        temp = 220
        ref_img_final = cv.imread(f"Correlable_pics/BOS_12_11_ref_masked ({temp}C).tif", cv.IMREAD_GRAYSCALE)
    else:
        temp = 252
        ref_img_final = cv.imread(f"Correlable_pics/BOS_12_11_ref_masked ({temp}C).tif", cv.IMREAD_GRAYSCALE)

    return work_img_final, ref_img_final, temp

# image_no = 2

# if image_no == 1 or image_no == 2:
#     mask_point = np.load(f"Mask_shapes/theBOSmask220C.npy")
# else:
#     mask_point = np.load(f"Mask_shapes/Mask_252.npy")

    
# mask_len = np.size(mask_point)
# mask_point = mask_point.reshape(int(mask_len/2),2)


# work_img = cv.imread(f"Raw_Pictures_Wavelet/BOS_12_11_{image_no}.tif")
# if image_no == 1 or image_no == 2:
#     #temperature of 220 degrees C
#     temp = 220
#     ref_img = cv.imread(f"Raw_Pictures_Wavelet/BOS_{temp}C_reference.tif")
# elif image_no == 3 or image_no == 4 or image_no == 5 or image_no == 6 or image_no == 7:
#     #temperature of 252 degrees C
#     temp = 252
#     ref_img = cv.imread(f"Raw_Pictures_Wavelet/BOS_{temp}C_reference.tif")
# else:
#     raise NameError("Image number not defined or invalid")

# ref_img = standard_pre(ref_img,1)
# work_img = standard_pre(work_img,1)

# mask = cv.polylines (ref_img, [mask_point], isClosed=True, color=(0, 0, 0), thickness=3)

# ref_img_M = cv.bitwise_and (ref_img, ref_img, mask=mask)
# work_img_M = cv.bitwise_and (work_img, work_img, mask=mask)

# ref_img_final = shape_isolation(ref_img,mask_point)
# work_img_final = shape_isolation(work_img,mask_point)

# # find the min and max y, to reduce the frames' dimensions
# max_y = np.max(mask_point[:,1])
# min_y = np.min(mask_point[:,1])

# # slice the picture
# ref_img_final = ref_img_final[min_y:max_y,:]
# work_img_final = work_img_final[min_y:max_y,:]

# # Save the masked images 
# cv.imwrite(f'Correlable_pics/BOS_12_11_{image_no}_masked.tif', work_img_final)
# cv.imwrite(f'Correlable_pics/BOS_12_11_ref_masked ({temp}C).tif', ref_img_final)


# # OPTIONAL visualize the masked images
# plt.subplot(1,2,1)
# plt.imshow(ref_img_final,cmap='gray')
# plt.subplot(1,2,2)
# plt.imshow(work_img_final,cmap='gray')
# plt.show()


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


def pixel_to_coords(x_pixels, y_pixels, throat_x, throat_y, SF=25.097):
    # SF = 25.097 [px/mm]
    # x_pixels = np.arange(x_pixels)
    # y_pixels = np.arange(y_pixels)

    x_pixels_grid, y_pixels_grid = np.meshgrid(
                                   np.arange(x_pixels), 
                                   np.arange(y_pixels))

    x_coords = (x_pixels_grid - throat_x) / SF
    y_coords = (y_pixels_grid - throat_y) / SF

    return x_coords, y_coords

image_no = 1
alpha = 35
blur = 15
blur_type = "gaussian"

work_img_final, ref_img_final, temp = get_images(image_no)
u = np.load(f"VF BOS_12_11_{image_no} ({temp}) corrected/u_HS_alpha{alpha}_blur{blur}_{blur_type}.npy")
v = np.load(f"VF BOS_12_11_{image_no} ({temp}) corrected/v_HS_alpha{alpha}_blur{blur}_{blur_type}.npy")

# scaling factor [px/mm]
SF = 25.097

u_reshaped, v_reshaped = reshape(u, v, ref_img_final)
u_reshaped = u_reshaped / SF
v_reshaped = v_reshaped / SF
throat_x, throat_y = find_throat_position(ref_img_final)
y_pixels, x_pixels = u_reshaped.shape
x_coords, y_coords = pixel_to_coords(x_pixels, y_pixels, throat_x, throat_y)


def build_dataframe(x, y, ux, uy):
    df = pd.DataFrame({
        "x": x.flatten(),
        "y": y.flatten(),
        "x-displacement": ux.flatten(),
        "y-displacement": uy.flatten()
    })
    return df

df = build_dataframe(x_coords, y_coords, u_reshaped, v_reshaped)
df.to_csv(f'OF_dataframes/BOS_12_11_{image_no} ({temp}C) df with alpha {alpha}, {blur_type} blur {blur}.csv', index=False)
print("Saved")