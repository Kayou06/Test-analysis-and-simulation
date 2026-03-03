def display_field():
    import numpy as np
    import cv2 as cv
    from OF_plot import draw_quiver

    u = np.load("u_HS.npy")
    v = np.load("v_HS.npy")

    beforeImg = cv.imread("Correlable_pics/BOS_12_11_ref_masked.tif", cv.IMREAD_GRAYSCALE)

    draw_quiver(u, v, beforeImg)


display_field()
