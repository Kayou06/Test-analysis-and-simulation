import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Read the data
df_BOS = pd.read_csv('BOS_12_11_10001.csv',delimiter=';') # BOS data
df_corr = pd.read_csv('cross_correction0001.csv',delimiter=';')  # correlation of crosses    

# get the average x displacement of the crosses
avg_x_corr = (df_corr
            #   .groupby('x-displacement')
              .agg(mean_x_corr = ('x-displacement','mean'))
)

# get the standard deviation of the x-displacement, and an average of the uncertainties on x-displacement of the crosses
std_x_corr = (df_corr
            #   .groupby('x-displacement')
              .agg(mean_x_unc = ('Uncertainty Vx','mean'),std_x_corr = ('x-displacement','std'))
              .reset_index()
)

# get the average y displacement of the crosses
avg_y_corr = (df_corr
            #   .groupby('x-displacement')
              .agg(mean_y_corr = ('y-displacement','mean'))
)

# get the standard deviation of the y-displacement, and an average of the uncertainties on y-displacement of the crosses
std_y_corr = (df_corr
            #   .groupby('x-displacement')
              .agg(mean_y_unc = ('Uncertainty Vy','mean'),std_y_corr = ('y-displacement','std'))
              .reset_index()
)

print(f"Mean x: {avg_x_corr}, std x: {std_x_corr}")
print(f"Mean y: {avg_y_corr}, std y: {std_y_corr}")

# use the correction data to correct the BOS data frame
df_BOS['x-displacement'] = df_BOS['x-displacement'] - avg_x_corr['x-displacement'][0]
df_BOS['y-displacement'] = df_BOS['y-displacement'] - avg_y_corr['y-displacement'][0]
df_BOS['Uncertainty Vx'] = np.sqrt(df_BOS['Uncertainty Vx']**2 + std_x_corr['Uncertainty Vx'][0]**2 + std_x_corr['x-displacement'][1]**2)
df_BOS['Uncertainty Vy'] = np.sqrt(df_BOS['Uncertainty Vy']**2 + std_y_corr['Uncertainty Vy'][0]**2 + std_y_corr['y-displacement'][1]**2)


df_BOS