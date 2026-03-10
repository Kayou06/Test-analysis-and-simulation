'''IMPORTS'''
import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np
# from Filters import *
from Pre_processing import standard_pre
# from Canny_visualizer import cannyEdge_visual
# from circle_finder import circles_finder
from video_maker import video_maker
from Masking import mask_points
from Masking import shape_isolation
from OF_plot import *
from Pyramidal_Horn_Schunck_tqdm import HS_pyramidal
# from blob_detector_function import cross_finder
# from cross_verification import match_score
from D02_cross_correction import cross_correction
from d02_display_field import *
from quick_plot import plot_midplane

root = os.getcwd()

'''CHOOSE the image number'''
image_no = 1

work_img = cv.imread(f"Raw_Pictures_Wavelet/BOS_12_11_{image_no}.tif")
if image_no == 1 or image_no == 2:
    #temperature of 220 degrees C
    ref_img = cv.imread("Raw_Pictures_Wavelet/BOS_220C_reference.tif")
else:
    #temperature of 252 degrees C
    ref_img = cv.imread("Raw_Pictures_Wavelet/BOS_252C_reference.tif")


'''Next parts contain optional plotting and visualization of images'''
# # OPTIONAL visualize the raw images

# plt.subplot(1,2,1)
# plt.imshow(ref_img,cmap='gray')
# plt.subplot(1,2,2)
# plt.imshow(work_img,cmap='gray')
# plt.show()

# # PRE-PROCESSING standard pre-processing applied:
# # ( scale (1 means no scaling), and histogram equalization)
ref_img = standard_pre(ref_img,1)
work_img = standard_pre(work_img,1)

# # OPTIONAL no pre-processing, this step is required to get a single channel
# # Only for other file formats than tiff
# ref_img = cv.cvtColor(ref_img, cv.COLOR_BGR2GRAY)
# work_img = cv.cvtColor(work_img, cv.COLOR_BGR2GRAY)

# # OPTIONAL visualize the normalized images
# plt.subplot(1,2,1)
# plt.imshow(ref_img,cmap='gray')
# plt.subplot(1,2,2)
# plt.imshow(work_img,cmap='gray')    
# plt.show()

# # OPTIONAL subtract the images to see the difference
# trial = work_img - ref_img
# plt.imshow(trial)
# plt.show()

'''Next parts contain mask configuration'''

# # Create a mask for the background region
# # For example, assuming the background is a specific color or can be segmented
# # Here, a dummy mask is created; replace this with your actual background mask
# background_mask = np.ones (ref_img.shape[:2], dtype=bool)

# # # If it's first run at particular conditions (i.e., BOS_x_y_z), use this function to create a mask
# # # The script will brake after the mask is created, but a npy file will be created

# mask_point = mask_points(ref_img,"BOS_12_11_1_mask.npy")

# # If a mask already exists, use this line, adjust the name based on the npy file created
mask_point = np.load("Mask_shapes/BOS_12_11_1_mask.npy")
mask_len = np.size(mask_point)
mask_point = mask_point.reshape(int(mask_len/2),2)

mask = cv.polylines (ref_img, [mask_point], isClosed=True, color=(0, 0, 0), thickness=3)

ref_img_M = cv.bitwise_and (ref_img, ref_img, mask=mask)
work_img_M = cv.bitwise_and (work_img, work_img, mask=mask)

ref_img_final = shape_isolation(ref_img,mask_point)
work_img_final = shape_isolation(work_img,mask_point)

# find the min and max y, to reduce the frames' dimensions
max_y = np.max(mask_point[:,1])
min_y = np.min(mask_point[:,1])

# slice the picture
ref_img_final = ref_img_final[min_y:max_y,:]
work_img_final = work_img_final[min_y:max_y,:]

# Save the masked images 
cv.imwrite('Correlable_pics/BOS_12_11_1_masked.tif', work_img_final)
cv.imwrite('Correlable_pics/BOS_12_11_ref_masked.tif', ref_img_final)

# OPTIONAL visualize the masked images
# plt.subplot(1,2,1)
# plt.imshow(ref_img_final,cmap='gray')
# plt.subplot(1,2,2)
# plt.imshow(work_img_final,cmap='gray')
# plt.show()

'''MAIN RUN'''

# # MAIN RUN
# # The number of levels is determined based on the maximum displacement expected
# # I keep 6 levels based on literature: https://doi.org/10.1007/s00348-022-03553-z 
# # The blur is based on the results from my Cross-Correlation pre-processubg
# # Alpha is based on some trial and error

'''CONFIGURE PARAMETERS'''
alpha = 35
blur =  11
blur_type = "gaussian"

# Compute and correct vector fields
u, v = HS_pyramidal(ref_img_final, work_img_final, alpha=alpha, levels=6, delta=1e-2, blr=blur, blur_type=blur_type)
u_corr, v_corr = cross_correction(u, v)

# Save vector fields
np.save(f"VF BOS_12_11_1 (220)/u_HS_alpha{alpha}_blur{blur}_{blur_type}.npy", u)
np.save(f"VF BOS_12_11_1 (220)/v_HS_alpha{alpha}_blur{blur}_{blur_type}.npy", v)
np.save(f"VF BOS_12_11_1 (220) corrected/u_HS_alpha{alpha}_blur{blur}_{blur_type}.npy", u)
np.save(f"VF BOS_12_11_1 (220) corrected/v_HS_alpha{alpha}_blur{blur}_{blur_type}.npy", v)

u_path = f"VF BOS_12_11_1 (220)/u_HS_alpha{alpha}_blur{blur}_{blur_type}.npy"
v_path = f"VF BOS_12_11_1 (220)/v_HS_alpha{alpha}_blur{blur}_{blur_type}.npy"
u = np.load(f"VF BOS_12_11_1 (220)/u_HS_alpha{alpha}_blur{blur}_{blur_type}.npy")
v = np.load(f"VF BOS_12_11_1 (220)/v_HS_alpha{alpha}_blur{blur}_{blur_type}.npy")



# # Visualize the results
# draw_quiver(u_corr,v_corr,ref_img_final)


plot_midplane(v,'original')
plot_midplane(v_corr,'corrected')
plt.legend()
plt.show()

display_many_fields(u_path, v_path, "u_corr.npy", "v_corr.npy")

# # Save the results
# np.save("u_HS", u)
# np.save("v_HS", v)
