'''IMPORTS'''
import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np
# from Filters import *
from Pre_processing import standard_pre
# from Canny_visualizer import cannyEdge_visual
# from circle_finder import circles_finder
from d02_display_field import display_many_fields
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
from Streamlinefunction_lower import streamline_lower
from Streamlinefunction_upper import streamline_upper
from Streamline_comparison import compare_streamlines
from d02_field_corrections import mask_correction
from pathlib import Path

root = os.getcwd()

if __name__ == "__main__":
    image_no = int(input("Enter the image number (1-7): "))

    streamline_upper(csv_path=f"CC Data/displacement_vectors{image_no}.csv", output_csv_path=f"CC_streamline/upper_results_{image_no}.csv")
    streamline_lower(csv_path=f"CC Data/displacement_vectors{image_no}.csv", output_csv_path=f"CC_streamline/lower_results_{image_no}.csv")
    compare_streamlines(upper_csv_path=f"CC_streamline/upper_results_{image_no}.csv", lower_csv_path=f"CC_streamline/lower_results_{image_no}.csv")

    '''CONFIGURE PARAMETERS'''
    alpha = 35
    blur =  11
    blur_type = "gaussian" #blur type is either "gaussian" or "median"


    work_img = cv.imread(f"Raw_Pictures_Wavelet/BOS_12_11_{image_no}.tif")
    if image_no == 1 or image_no == 2:
        #temperature of 220 degrees C
        temp = 220
        ref_img = cv.imread(f"Raw_Pictures_Wavelet/BOS_{temp}C_reference.tif")
    elif image_no == 3 or image_no == 4 or image_no == 5 or image_no == 6 or image_no == 7:
        #temperature of 252 degrees C
        temp = 252
        ref_img = cv.imread(f"Raw_Pictures_Wavelet/BOS_{temp}C_reference.tif")
    else:
        raise NameError("Image number not defined or invalid")


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
    if image_no == 1 or image_no == 2:
        mask_point = np.load(f"Mask_shapes/theBOSmask220C.npy")
    else:
        mask_point = np.load(f"Mask_shapes/Mask_252.npy")

    
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
    cv.imwrite(f'Correlable_pics/BOS_12_11_{image_no}_masked.tif', work_img_final)
    cv.imwrite(f'Correlable_pics/BOS_12_11_ref_masked ({temp}C).tif', ref_img_final)

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

    '''Either compute a NEW vector field or load an EXISTING vector field'''

    file1 = Path(f"VF BOS_12_11_{image_no} ({temp})/u_HS_alpha{alpha}_blur{blur}_{blur_type}.npy")
    file2 = Path(f"VF BOS_12_11_{image_no} ({temp})/v_HS_alpha{alpha}_blur{blur}_{blur_type}.npy")
    file3 = Path(f"VF BOS_12_11_{image_no} ({temp}) corrected/u_HS_alpha{alpha}_blur{blur}_{blur_type}.npy")
    file4 = Path(f"VF BOS_12_11_{image_no} ({temp}) corrected/v_HS_alpha{alpha}_blur{blur}_{blur_type}.npy")

    if file1.exists() and file2.exists() and file3.exists() and file4.exists():
        # Load already existing vector fields
        u = np.load(file1)
        v = np.load(file2)
        u_corr = np.load(file3)
        v_corr = np.load(file4)
    else:
        # Compute new uncorrected and corrected vector fields
        u, v = HS_pyramidal(ref_img_final, work_img_final, alpha=alpha, levels=6, delta=1e-2, blr=blur, blur_type=blur_type)
        u_corr, v_corr = cross_correction(u, v, picture_no=image_no)
        # Save vector fields
        np.save(file1, u)
        np.save(file2, v)
        np.save(file3, u)
        np.save(file4, v)


    u_corr, v_corr = mask_correction(u_corr, v_corr, mask_point)


    '''NEXT SECTION IS FOR CROSS-CORRELATION METHOD'''



    '''NEXT SECTION IS FOR VISUALIZING RESULTS'''

    # draw_quiver(u_corr,v_corr,ref_img_final)

    plot_midplane(v,'original')
    plot_midplane(v_corr,'corrected')
    plt.legend()
    plt.show()

   
    
    # Use display_many_fields function to plot vector field in the nozzle
    display_many_fields_object([(u, v, ref_img_final, f"Uncorrected vector field at alpha = {alpha}, blur = {blur}"),
                                (u_corr, v_corr, ref_img_final, f"Corrected vector field at alpha = {alpha}, blur = {blur}")])