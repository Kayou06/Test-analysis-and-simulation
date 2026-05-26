import numpy as np
import math

def mask_correction(u, v, mask):
    # Index i is the y axis in the graph. Index j is the x axis in the graph.
    scale = 10 # This is the same as when drawing the quiver plot.
    for i, row in enumerate(mask):
        for j, pixel in enumerate(row):
            if mask[i][j] == 255:
                u[i][j] = 0
                v[i][j] = 0
            
            new_i = i + math.ceil(v[i][j]*scale)
            new_j = j + math.ceil(u[i][j]*scale)

            #new_i = i + math.ceil(v[i][j])
            #new_j = j + math.ceil(u[i][j])

            try:   
                if mask[new_i][new_j] == 255:
                    u[i][j] = 0
                    v[i][j] = 0
            except IndexError:
                pass

    return u, v

def mask_correction_densitygrad(density_gradient, mask):
    scale = 10
    mask_flat = mask.flatten()

    # Copy to avoid modifying original
    density_gradient_corrected = density_gradient.copy()

    # Set values outside mask to zero
    density_gradient_corrected[mask_flat == 255] = 0

    return density_gradient_corrected