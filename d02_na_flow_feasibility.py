import numpy as np
import cv2 as cv
import math

def remove_vertical(u, v, region: list, angle):
    lower_bound_u = region[0][0]
    upper_bound_u = region[0][1]   
    lower_bound_v = region[1][0]
    upper_bound_v = region[1][1]
    print(f"Region bounds: u in [{lower_bound_u}, {upper_bound_u}], v in [{lower_bound_v}, {upper_bound_v}]")
    count = 0

    for val_u in range(lower_bound_u, upper_bound_u):
        for val_v in range(lower_bound_v, upper_bound_v):
           if abs(math.atan(v[val_u][val_v]/u[val_u][val_v])) > angle:
                u[val_u][val_v] = 0
                v[val_u][val_v] = 0
                count += 1

    return u, v, count
