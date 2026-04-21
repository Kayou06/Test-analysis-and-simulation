import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from Compressibility import compressibility_factor_and_density
from D02_CC_densitygrad import calc_drho_dx

Z, rho0 = compressibility_factor_and_density()

def throat_position():

    return x_pixel_throat, y_pixel_throat

def build_dataframe():


    OF_df = pd.DataFrame({
        'x':
        'y':
        'x_displacement':
        'y_displacement':

    })


    return OF_df


def extract_midline_displacement_OF():
    a=1



def plot_OF_densitygrad(image_no, alpha, blur, blur_type):
    rho = rho0[image_no - 1]

    midline_df = extract_midline_displacement_OF()
    drho_dx = calc_drho_dx(midline_df['x_displacement'], rho)

    closest_idx = np.abs(midline_df['x']).argmin()
    drho_dx_at_x0 = drho_dx.iloc[closest_idx]

    normalized_drho_dx = drho_dx / drho_dx_at_x0