import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from D02_density_gradient import epsilon


#Select the correct file path
df_BOS = pd.read_csv('Raw_Pictures_Wavelet/BOS_12_11_7/BOS_12_11_70001.csv',delimiter=';') # BOS data
df_corr = pd.read_csv('Raw_Pictures_Wavelet/BOS_12_11_7/cross_correction0001.csv',delimiter=';')  # correlation of crosses     

# Cross Correlation Plot
x = df_BOS['x']
y = df_BOS['y']
u = df_BOS['x-displacement']
v = df_BOS['y-displacement']

#corrected Plot
u_corr = df_corr['x-displacement'].mean()
v_corr = df_corr['y-displacement'].mean()


#final displacements
u_final = u - u_corr
v_final = v - v_corr


# Calculate the speed (magnitude of the displacement vector)
speed = np.sqrt(u_final**2 + v_final**2)

# Create vector field plot 1 

plt.figure(figsize=(8,6))
plt.quiver(x, y, u_final, v_final, speed, scale_units='xy', scale = 0.25)
plt.colorbar(label='Displacement Magnitude')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Displacement Vector Field")
plt.axis('equal')
plt.grid(True)


#Saves file to a folder named "CC data" in the current working directory. The file will contain the x and y coordinates along with the corrected displacement vectors (u_final and v_final) for each point in the BOS data.
np.save(F"CC data/displacement_vectors.npy", np.column_stack((x, y, u_final, v_final)))

plt.show()

y_mid = np.mean(y)

midline_mask = (y == y_mid)
# Extract midline values
x_mid = x[midline_mask]
u_mid = u_final[midline_mask]
v_mid = v_final[midline_mask]

# Sort from left to right
sort_idx = np.argsort(x_mid)
x_mid = x_mid[sort_idx]
u_mid = u_mid[sort_idx]
v_mid = v_mid[sort_idx]

# Save midline displacements
midline_df = pd.DataFrame({
    'x': x_mid,
    'x_displacement': u_mid,
    'y_displacement': v_mid
})
midline_df.to_csv("CC data/midline_displacements.csv", index=False)

def epsilon(del_x):
    C = 5.3
    ZD = 10
    ZA = 1200+40+10
    f = 200
    eps = del_x*(ZD+ZA+ZA-f)/(C*f*ZD)

    return eps

def normalized_epsilon(del_x, del_x_1):
    eps = epsilon(del_x)
    eps_max = epsilon(del_x_1)
    return eps / eps_max

midline_df['epsilon'] = midline_df['x_displacement'].apply(epsilon)

eps_at_x0 = midline_df.loc[midline_df['x'] == 0, 'epsilon'].values[0]

midline_df['normalized_epsilon'] = midline_df['epsilon'] / eps_at_x0