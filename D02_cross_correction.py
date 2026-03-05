import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#scaling factor from READ ME.txt
SF = 25.097 #[px/mm]

def cross_correction(u, v):
# read our own BOS data
# u = np.load("VF BOS_12_11_1 (220)/u_HS_alpha25_blur5.npy")
# v = np.load("VF BOS_12_11_1 (220)/v_HS_alpha25_blur5.npy")

    # read tutor's cross-corretion data - DEPENDS ON IMAGE
    df_corr = pd.read_csv('Raw_Pictures_Wavelet/BOS_12_11_1/cross_correction0001.csv',delimiter=';')      

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
    u_corr = u - avg_x_corr.iloc[0,0]
    v_corr = v - avg_y_corr.iloc[0,0]

    return u_corr, v_corr