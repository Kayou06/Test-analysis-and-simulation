import cv2 as cv
import numpy as np
from Pre_processing import standard_pre
from Masking import shape_isolation

# This file contains a single function which performs preprocessing for OF flow images:
# it uses standard pre-processing function, masks and crops the images.

def image_preprocessing(image_no):

    # # Create a mask for the background region
    # # For example, assuming the background is a specific color or can be segmented
    # # Here, a dummy mask is created; replace this with your actual background mask
    # background_mask = np.ones (ref_img.shape[:2], dtype=bool)

    # # If it's first run at particular conditions (i.e., BOS_x_y_z), use this function to create a mask
    # # The script will brake after the mask is created, but a npy file will be created

    # mask_point = mask_points(ref_img,"BOS_12_11_1_mask.npy")

    # # If a mask already exists, use this line, adjust the name based on the npy file created
    if image_no == 1 or image_no == 2:
        mask_point = np.load(f"Mask_shapes/theBOSmask220C.npy")
    else:
        mask_point = np.load(f"Mask_shapes/Mask_252.npy")
    
    mask_len = np.size(mask_point)
    mask_point = mask_point.reshape(int(mask_len/2),2)
        
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
    
    # # PRE-PROCESSING standard pre-processing applied:
    # # ( scale (1 means no scaling), and histogram equalization)
    ref_img = standard_pre(ref_img,1)
    work_img = standard_pre(work_img,1)

    # # OPTIONAL no pre-processing, this step is required to get a single channel
    # # Only for other file formats than tiff
    # ref_img = cv.cvtColor(ref_img, cv.COLOR_BGR2GRAY)
    # work_img = cv.cvtColor(work_img, cv.COLOR_BGR2GRAY)

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

    return ref_img_final, work_img_final, temp
