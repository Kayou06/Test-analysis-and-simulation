import numpy as np
import cv2 as cv
from d02_display_field import display_many_fields
import math

u = np.load("u_HS.npy")
v = np.load("v_HS.npy") # V is upwards, U is rightwards
mask = cv.imread("Correlable_pics/BOS_12_11_ref_masked.tif", cv.IMREAD_GRAYSCALE)

k = 0
for i, row in enumerate(mask):
    for j, pixel in enumerate(row):
        if mask[i][j] == 255:
            u[i][j] = 0
            v[i][j] = 0
        
        new_i = i + math.ceil(v[i][j])
        new_j = j + math.ceil(u[i][j])

        if mask[new_i][new_j] == 255:
            u[i][j] = 0
            v[i][j] = 0
            k+=1


# Index i is the y axis in the graph. Index j is the x axis in the graph.

np.save("u_corr", u)
np.save("v_corr", v)

u_or = np.load("u_HS.npy")
v_or = np.load("v_HS.npy")

display_many_fields([("u_corr.npy", "v_corr.npy", "Correlable_pics/BOS_12_11_ref_masked.tif", "Corrected Field"), ("u_HS.npy", "v_HS.npy", "Correlable_pics/BOS_12_11_ref_masked.tif", "Original Field")])

print(k)
