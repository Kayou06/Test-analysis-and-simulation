import numpy as np
import math

def mask_correction(u, v, mask):
    # Index i is the y axis in the graph. Index j is the x axis in the graph.
    for i, row in enumerate(mask):
        for j, pixel in enumerate(row):
            if mask[i][j] == 255:
                u[i][j] = 0
                v[i][j] = 0
            
            new_i = i + math.ceil(v[i][j]*10)
            new_j = j + math.ceil(u[i][j]*10)

            try:   
                if mask[new_i][new_j] == 255:
                    u[i][j] = 0
                    v[i][j] = 0
            except IndexError:
                pass

    np.save("u_corr", u)
    np.save("v_corr", v)

