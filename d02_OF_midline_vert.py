import numpy as np

def find_midline_y(u, v):
    v_bounded = v[212:412 + 1, :]
    v_sum = np.sum(v_bounded, axis=1)

    vertical_score = np.abs(v_sum)

    midline_y_index = int(np.argmax(vertical_score)) + 212

    return midline_y_index


u = np.load("u_HS.npy")
v = np.load("v_HS.npy")

midline_y_index = find_midline_y(u, v)

print(f"Midline y-index: {midline_y_index}")
print(f"Shape of u: {u.shape}, Shape of v: {v.shape}")
