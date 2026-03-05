import numpy as np
import cv2 as cv
from OF_plot import draw_quiver, draw_quiver_many

# Prety much just a file to display the results of the optical flow calculations, using the quiver plots from OF_plot.py
# Self explantory, you can add or remove as many vector fields as you want
def display_field():
    u = np.load("u_HS.npy")
    v = np.load("v_HS.npy")

    beforeImg = cv.imread("Correlable_pics/BOS_12_11_ref_masked.tif", cv.IMREAD_GRAYSCALE)

    draw_quiver(u, v, beforeImg)


def display_many_fields(u1_path, v1_path, u2_path, v2_path):

    u1 = np.load(u1_path)
    v1 = np.load(v1_path)
    img1 = cv.imread("Correlable_pics/BOS_12_11_ref_masked.tif", cv.IMREAD_GRAYSCALE)

    u2 = np.load(u2_path)
    v2 = np.load(v2_path)
    img2 = cv.imread("Correlable_pics/BOS_12_11_ref_masked.tif", cv.IMREAD_GRAYSCALE)

    draw_quiver_many(
        [u1, u2],
        [v1, v2],
        [img1, img2],
        titles=["Run 1", "Run 2"]
    )


if __name__ == "__main__":
    display_many_fields()
