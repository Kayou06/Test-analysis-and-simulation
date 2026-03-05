import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


#compute magnitude in each 8 pixels. return magnitude average
def get_magnitude(u, v):
    scale = 1
    sum = 0.0
    counter = 0.0

    for i in range(0, u.shape[0], 8):
        for j in range(0, u.shape[1],8):
            counter += 1
            dy = v[i,j] 
            dx = u[i,j] 
            magnitude = (dx**2 + dy**2)**0.5
            sum += magnitude

    mag_avg = sum / counter

    return mag_avg



def draw_quiver(u,v,beforeImg):
    scale = 10
    ax = plt.figure().gca()
    ax.imshow(beforeImg, cmap = 'gray')

    magnitudeAvg = get_magnitude(u, v)

    for i in range(0, u.shape[0], 8):
        for j in range(0, u.shape[1],8):
            dy = v[i,j] 
            dx = u[i,j] 
            magnitude = (dx**2 + dy**2)**0.5
            #draw only significant changes
            if magnitude > magnitudeAvg:
                ax.arrow(j,i, dx*scale, dy*scale, color = 'red')

    plt.draw()
    plt.show()


# Fancy matplotlib stuff to display multiple plots at once
def draw_quiver_many(u, v, beforeImg, titles=None, *, scale=10, step=8):
    many = isinstance(u, (list, tuple))

    if not many:
        us = [u]
        vs = [v]
        imgs = [beforeImg]
    else:
        us = list(u)
        vs = list(v)
        imgs = list(beforeImg)

    n = len(us)
    if len(vs) != n or len(imgs) != n:
        raise ValueError("u, v, beforeImg must have the same number of items")

    if titles is None:
        titles_list = None
    elif isinstance(titles, str):
        titles_list = [titles] + [None] * (n - 1)
    else:
        if len(titles) != n:
            raise ValueError("titles must have the same length as the number of fields")
        titles_list = list(titles)

    fig, axs = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axs = [axs]

    for k in range(n):
        ax = axs[k]
        uu = us[k]
        vv = vs[k]
        img = imgs[k]

        ax.imshow(img, cmap="gray")

        magnitudeAvg = get_magnitude(uu, vv)

        for i in range(0, uu.shape[0], step):
            for j in range(0, uu.shape[1], step):
                dy = vv[i, j]
                dx = uu[i, j]
                magnitude = (dx**2 + dy**2)**0.5
                if magnitude > magnitudeAvg:
                    ax.arrow(j, i, dx * scale, dy * scale, color="red")

        if titles_list is not None and titles_list[k] is not None:
            ax.set_title(titles_list[k])

        ax.set_axis_off()

    plt.tight_layout()
    plt.show()
