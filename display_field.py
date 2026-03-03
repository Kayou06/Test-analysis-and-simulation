import numpy as np
import cv2 as cv
from OF_plot import draw_quiver, draw_quiver_many

def display_field():
    u = np.load("u_HS.npy")
    v = np.load("v_HS.npy")

    beforeImg = cv.imread("Correlable_pics/BOS_12_11_ref_masked.tif", cv.IMREAD_GRAYSCALE)

    draw_quiver(u, v, beforeImg)


def display_many_fields():
    u1 = np.load("u_HS.npy")
    v1 = np.load("v_HS.npy")
    img1 = cv.imread("Correlable_pics/BOS_12_11_ref_masked.tif", cv.IMREAD_GRAYSCALE)

    u2 = np.load("u_HS_alpha10_blur1.npy")
    v2 = np.load("v_HS_alpha10_blur1.npy")
    img2 = cv.imread("Correlable_pics/BOS_12_11_ref_masked.tif", cv.IMREAD_GRAYSCALE)

    draw_quiver_many(
        [u1, u2],
        [v1, v2],
        [img1, img2],
        titles=["Run 1", "Run 2"]
    )


if __name__ == "__main__":
    display_many_fields()
