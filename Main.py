'''IMPORTS'''
import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np
from OF_plot import *
from Pyramidal_Horn_Schunck_tqdm import HS_pyramidal, reshape
from D02_cross_correction import cross_correction
from d02_display_field import *
from quick_plot import plot_midplane
from d02_field_corrections import mask_correction
from D02_data_preprocessing import image_preprocessing
from D02_data_postprocessing import data_postprocessing, build_dataframe
# from D02_OF_densitygrad import pixel_to_coords, build_dataframe
# from D02_Streamlinefunction_lower import streamline_lower
# from D02_Streamlinefunction_upper import streamline_upper
# from D02_Streamline_comparison import compare_streamlines
# from video_maker import video_maker
# from Masking import mask_points
# from Masking import shape_isolation
# from d02_display_field import display_many_fields
# from Pre_processing import standard_pre
# from blob_detector_function import cross_finder
# from cross_verification import match_score
# from Filters import *
# from Canny_visualizer import cannyEdge_visual
# from circle_finder import circles_finder

# Highest to lowest compressibility factor:
# 7 - 1 - 6 - 2 & 5 - 3 - 4

root = os.getcwd()


if __name__ == "__main__":
    image_no = int(input("Enter the image number (1-7): "))

    '''CONFIGURE PARAMETERS'''
    alpha = 35
    blur =  11
    blur_type = "gaussian" #blur type is either "gaussian" or "median"

    # # PRE-PROCESSING
    ref_img_final, work_img_final, temp = image_preprocessing(image_no)

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
        print("Files loaded successfully.")
    else:
        # Compute new uncorrected AND cross-corrected vector fields
        print(f"Trying to compute new vector fields for image {image_no} with alpha {alpha}, blur {blur}, and blur type {blur_type}.")
        u, v = HS_pyramidal(ref_img_final, work_img_final, alpha=alpha, levels=6, delta=1e-2, blr=blur, blur_type=blur_type)
        u_corr, v_corr = cross_correction(u, v, picture_no=image_no)
        # Save vector fields
        np.save(file1, u)
        np.save(file2, v)
        np.save(file3, u_corr)
        np.save(file4, v_corr)
        print("Files computed, corrected and saved successfully.")

    # Apply post-processing steps to cross-corrected vector field
    x_coords, y_coords, u_final, v_final = data_postprocessing(image_no, u_corr, v_corr)
    df = build_dataframe(x_coords, y_coords, u_final, v_final)
    df.to_csv(f'OF_dataframes/BOS_12_11_{image_no} ({temp}C) df with alpha {alpha}, {blur_type} blur {blur}.csv', index=False)
    print("Succesfully saved final dataframe.")

    '''VISUALIZING RESULTS'''

    # # Apply reshaping and masking, only for plotting!!
    # u_plot, v_plot = reshape(u, v, ref_img_final)
    # u_plot, v_plot = mask_correction(u_plot, v_plot, ref_img_final)
    # u_corr_plot, v_corr_plot = reshape(u_corr, v_corr, ref_img_final)
    # u_corr_plot, v_corr_plot = mask_correction(u_corr_plot, v_corr_plot, ref_img_final)

    # plot_midplane(v,'original')
    # plot_midplane(v_corr,'corrected')
    # plt.legend()
    # plt.show()

    # # draw_quiver(u_corr,v_corr,ref_img_final)

    # # streamline_upper(csv_path=f"CC Data/displacement_vectors{image_no}.csv")
    # # streamline_lower(csv_path=f"CC Data/displacement_vectors{image_no}.csv")
    # # compare_streamlines(upper_csv_path=f"CC_streamline/upper_results_{image_no}.csv", lower_csv_path=f"CC_streamline/lower_results_{image_no}.csv")
    
    # # Use display_many_fields function to plot vector field in the nozzle
    # display_many_fields_object([(u_plot, v_plot, ref_img_final, f"Uncorrected vector field at alpha = {alpha}, blur = {blur}"),
    #                             (u_corr_plot, v_corr_plot, ref_img_final, f"Corrected vector field at alpha = {alpha}, blur = {blur}")])
    