import numpy as np
import cv2 as cv
from OF_plot import draw_quiver, draw_quiver_many

def display_field():
    u = np.load("u_HS.npy")
    v = np.load("v_HS.npy")

    beforeImg = cv.imread("Correlable_pics/BOS_12_11_ref_masked.tif", cv.IMREAD_GRAYSCALE)

    draw_quiver(u, v, beforeImg)


def display_many_fields():
    root = "VF BOS_12_11_1 (220)/"

    u1 = np.load(root + "u_HS_alpha5_blur5.npy")
    v1 = np.load(root + "v_HS_alpha5_blur5.npy")
    img1 = cv.imread("Correlable_pics/BOS_12_11_ref_masked.tif", cv.IMREAD_GRAYSCALE)

    u2 = np.load(root + "u_HS_alpha10_blur5.npy")
    v2 = np.load(root + "v_HS_alpha10_blur5.npy")
    img2 = cv.imread("Correlable_pics/BOS_12_11_ref_masked.tif", cv.IMREAD_GRAYSCALE)

    u3 = np.load(root + "u_HS_alpha15_blur5.npy")
    v3 = np.load(root + "v_HS_alpha15_blur5.npy")
    img3 = cv.imread("Correlable_pics/BOS_12_11_ref_masked.tif", cv.IMREAD_GRAYSCALE)

    u4 = np.load(root + "u_HS_alpha20_blur5.npy")
    v4 = np.load(root + "v_HS_alpha20_blur5.npy")
    img4 = cv.imread("Correlable_pics/BOS_12_11_ref_masked.tif", cv.IMREAD_GRAYSCALE)

    u5 = np.load(root + "u_HS_alpha25_blur5.npy")
    v5 = np.load(root + "v_HS_alpha25_blur5.npy")
    img5 = cv.imread("Correlable_pics/BOS_12_11_ref_masked.tif", cv.IMREAD_GRAYSCALE)

    u6 = np.load(root + "u_HS_alpha30_blur5.npy")
    v6 = np.load(root + "v_HS_alpha30_blur5.npy")
    img6 = cv.imread("Correlable_pics/BOS_12_11_ref_masked.tif", cv.IMREAD_GRAYSCALE)

    u7 = np.load(root + "u_HS_alpha35_blur5.npy")
    v7 = np.load(root + "v_HS_alpha35_blur5.npy")
    img7 = cv.imread("Correlable_pics/BOS_12_11_ref_masked.tif", cv.IMREAD_GRAYSCALE)


    draw_quiver_many(
        [u1, u2, u3, u4, u5, u6, u7],
        [v1, v2, v3, v4, v5, v6, v7],
        [img1, img2, img3, img4, img5, img6, img7],
        titles=["Run 1", "Run 2", "Run 3", "Run 4", "Run 5", "Run 6", "Run 7"]
    )


if __name__ == "__main__":
    display_many_fields()
