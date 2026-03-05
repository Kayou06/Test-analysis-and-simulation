import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#scaling factor from READ ME.txt
SF = 25.097 #[px/mm]

# read our own BOS data
df_BOS = pd.read_csv('BOS_12_11_10001.csv',delimiter=';') 

# read tutor's cross-corretion data - DEPENDS ON IMAGE
df_corr = pd.read_csv('cross_correction0001.csv',delimiter=';')      

# get the average x displacement of the crosses
avg_x_corr = (df_corr
            #   .groupby('x-displacement')
              .agg(mean_x_corr = ('x-displacement','mean'))
)

# get the average y displacement of the crosses
avg_y_corr = (df_corr
            #   .groupby('x-displacement')
              .agg(mean_y_corr = ('y-displacement','mean'))
)

print(f"Mean x: {avg_x_corr}")
print(f"Mean y: {avg_y_corr}")

# use the correction data to correct the BOS data frame
df_BOS['x-displacement'] = df_BOS['x-displacement'] - avg_x_corr['x-displacement'][0]
df_BOS['y-displacement'] = df_BOS['y-displacement'] - avg_y_corr['y-displacement'][0]

df_BOS