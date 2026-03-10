import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#scaling factor from READ ME.txt
SF = 25.097 #[px/mm]

def cross_correction(u, v, picture_no=1):

    # read tutor's cross-corretion data - DEPENDS ON IMAGE
    df_corr = pd.read_csv(f'Raw_Pictures_Wavelet/BOS_12_11_{picture_no}/cross_correction0001.csv',delimiter=';')      

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
    u_corr = u - avg_x_corr.iloc[0,0] * SF
    v_corr = v - avg_y_corr.iloc[0,0] * SF

    return u_corr, v_corr