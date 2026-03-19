import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Select the correct file path
df_BOS = pd.read_csv('Raw_Pictures_Wavelet/BOS_12_11_7/BOS_12_11_70001.csv', delimiter=';')
df_corr = pd.read_csv('Raw_Pictures_Wavelet/BOS_12_11_7/cross_correction0001.csv', delimiter=';')

# Cross Correlation Plot
x = df_BOS['x']
y = df_BOS['y']
u = df_BOS['x-displacement']
v = df_BOS['y-displacement']

# Corrected Plot
u_corr = df_corr['x-displacement'].mean()
v_corr = df_corr['y-displacement'].mean()

# Final displacements
u_final = u - u_corr
v_final = v - v_corr

# Calculate the speed (magnitude of the displacement vector)
speed = np.sqrt(u_final**2 + v_final**2)

# Create vector field plot
plt.figure(figsize=(8, 6))
plt.quiver(x, y, u_final, v_final, speed, scale_units='xy', scale=0.25)
plt.colorbar(label='Displacement Magnitude')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Displacement Vector Field")
plt.axis('equal')
plt.grid(True)

np.save("CC data/displacement_vectors.npy", np.column_stack((x, y, u_final, v_final)))
plt.show()

# Midline extraction
y_mid = np.mean(y)
midline_mask = (y == y_mid)

x_mid = x[midline_mask]
u_mid = u_final[midline_mask]
v_mid = v_final[midline_mask]

# Sort from left to right
sort_idx = np.argsort(x_mid)
x_mid = x_mid.values[sort_idx]
u_mid = u_mid.values[sort_idx]
v_mid = v_mid.values[sort_idx]

# Save midline displacements
midline_df = pd.DataFrame({
    'x': x_mid,
    'x_displacement': u_mid,
    'y_displacement': v_mid
})
midline_df.to_csv("CC data/midline_displacements.csv", index=False)

# FIX 1: Rename internal variable to avoid conflict with function name
def calc_drho_dx(del_x):
    C = 5.3
    ZD = 0.010
    ZA = 1.250
    f = 0.200
    W = 0.020
    K = 4.5*10**-4
    n0 = 1.000293
    result = del_x * n0 * (ZD + ZA - f) / (C * W * K * f * ZD)
    return result

midline_df['drho_dx'] = midline_df['x_displacement'].apply(calc_drho_dx)

# FIX 2: Extract scalar with .values[0], fall back to nearest if x=0 not found
drho_dx_at_x0 = midline_df.loc[midline_df['x'] == 0, 'drho_dx'].values[0]


midline_df['normalized_drho_dx'] = midline_df['drho_dx'] / drho_dx_at_x0

# Plot
plt.figure()
plt.plot(midline_df['x'], midline_df['normalized_drho_dx'])
plt.xlabel('x')
plt.ylabel('Normalized drho/dx')
plt.title('Normalized drho/dx vs x')
plt.grid(True)
plt.show()