import numpy as np
import cv2 as cv
from pathlib import Path
from OF_plot import draw_quiver, draw_quiver_many

# Prety much just a file to display the results of the optical flow calculations, using the quiver plots from OF_plot.py
# Self explantory, you can add or remove as many vector fields as you want
def display_field():
    u = np.load("u_HS.npy")
    v = np.load("v_HS.npy")

    beforeImg = cv.imread("Correlable_pics/BOS_12_11_ref_masked.tif", cv.IMREAD_GRAYSCALE)

    draw_quiver(u, v, beforeImg)


def display_many_fields(runs, *, titles=None, img_flag=cv.IMREAD_GRAYSCALE):
    us, vs, imgs = [], [], []
    auto_titles = []

    for i, run in enumerate(runs, start=1):
        if len(run) == 3:
            u_p, v_p, img_p = run
            t = None
        elif len(run) == 4:
            u_p, v_p, img_p, t = run
        else:
            raise ValueError(f"Each run must have 3 or 4 items, got {len(run)}: {run}")

        u_path = Path(u_p)
        v_path = Path(v_p)
        img_path = Path(img_p)

        us.append(np.load(u_path))
        vs.append(np.load(v_path))

        img = cv.imread(str(img_path), img_flag)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        imgs.append(img)

        auto_titles.append(t if t is not None else f"Run {i}")

    if titles is None:
        titles = auto_titles
    elif len(titles) != len(us):
        raise ValueError(f"titles has length {len(titles)} but there are {len(us)} runs")

    draw_quiver_many(us, vs, imgs, titles=titles)


'''
How to use this function, title can also be empty. Image flag can be changed accordingly as well.

runs = [
    ("VF BOS_12_11_1 (220)/u_HS_alpha5_blur5.npy",
     "VF BOS_12_11_1 (220)/v_HS_alpha5_blur5.npy",
     "Correlable_pics/BOS_12_11_ref_masked.tif",
     "Run 1"),

    ("VF BOS_12_11_1 (220)/u_HS_alpha10_blur5.npy",
     "VF BOS_12_11_1 (220)/v_HS_alpha10_blur5.npy",
     "Correlable_pics/BOS_12_11_ref_masked.tif",
     "Run 2"),

     ("VF BOS_12_11_1 (220)/u_HS_alpha15_blur5.npy",
     "VF BOS_12_11_1 (220)/v_HS_alpha15_blur5.npy",
     "Correlable_pics/BOS_12_11_ref_masked.tif",
     "Run 3")
]

display_many_fields(runs)
'''

def display_many_fields_object(runs, *, titles=None):
    us, vs, masks = [], [], []
    auto_titles = []

    for i, run in enumerate(runs, start=1):
        if len(run) == 3:
            u, v, mask = run
            t = None
        elif len(run) == 4:
            u, v, mask, t = run
        else:
            raise ValueError(f"Each run must have 3 or 4 items, got {len(run)}: {run}")

        us.append(np.asarray(u))
        vs.append(np.asarray(v))
        masks.append(np.asarray(mask))

        auto_titles.append(t if t is not None else f"Run {i}")

    if titles is None:
        titles = auto_titles
    elif len(titles) != len(us):
        raise ValueError(f"titles has length {len(titles)} but there are {len(us)} runs")

    draw_quiver_many(us, vs, masks, titles=titles)

# The one above takes the actual u, v, and mask objects instead of the paths to the files.
# This is useful if you want to do some corrections to the fields before displaying them, without having to save them as npy files first.
