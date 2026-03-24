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
y_mid = y.iloc[(y - 0).abs().argmin()]
midline_mask = np.isclose(y, y_mid)

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

#test
print(midline_df)



midline_df.to_csv("CC data/midline_displacements.csv", index=False)
rho0 = [40.27772187279685, 75.04027767585774, 139.0060691158923, 209.16092353098253, 91.60892909471781, 59.861281984102675, 41.428026313530424]


# FIX 1: Rename internal variable to avoid conflict with function name
def calc_drho_dx(del_x, rho):
    C = 5.3
    ZD = 0.010
    ZA = 1.250
    f = 0.200
    W = 0.020
    K = 4.5*10**-4
    n0 = K*rho+1
    result = del_x * n0 * (ZD + ZA - f) / (C * W * K * f * ZD)
    return result


for rho in rho0:
    # FIX 1: Call function directly on the Series (no .apply needed)
    drho_dx = calc_drho_dx(midline_df['x_displacement'], rho)
    midline_df['drho_dx'] = drho_dx


    plt.plot(midline_df['x'], midline_df['drho_dx'], label=f'rho0={rho:.1f}')

plt.xlabel('x')
plt.ylabel('drho/dx')
plt.title('drho/dx vs x')
plt.grid(True)
plt.show()

plt.figure()

for rho in rho0:
    # FIX 1: Call function directly on the Series (no .apply needed)
    drho_dx = calc_drho_dx(midline_df['x_displacement'], rho)
    midline_df['drho_dx'] = drho_dx

    # FIX 2: Use np.isclose to safely find the row nearest to x=0
    closest_idx = (midline_df['x'] - 0).abs().argmin()
    drho_dx_at_x0 = midline_df.loc[closest_idx, 'drho_dx']  # scalar

    # FIX 3: Divide by scalar directly
    midline_df['normalized_drho_dx'] = midline_df['drho_dx'] / drho_dx_at_x0

    plt.plot(midline_df['x'], midline_df['normalized_drho_dx'], label=f'rho0={rho:.1f}')
plt.xlabel('x')
plt.ylabel('Normalized drho/dx')
plt.title('Normalized drho/dx vs x')
plt.grid(True)
plt.show()