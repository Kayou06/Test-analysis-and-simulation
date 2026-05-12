import cv2 as cv
import numpy as np
from circle_finder import circles_finder

def find_midline(path="Raw_Pictures_Wavelet/BOS_12_11_1.tif"):
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)

    circles = circles_finder(
        img=img,
        blur_lvl=5,
        xmin=700,
        xmax=1050,
        r_t=100,
        r_w=100
    )

    p1 = np.array(circles[0])
    p2 = np.array(circles[1])

    return ((p1[:2].astype(float) + p2[:2].astype(float)) / 2 )


if "__main__" == __name__:
    print(find_midline("Raw_Pictures_Wavelet/BOS_12_11_1.tif"))
