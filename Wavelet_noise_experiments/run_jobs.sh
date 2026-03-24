#!/bin/bash

# Base directory
BASE_DIR="/media/assecchiaroli/Data2/BOS_VV_Dec_2025_CFD/Wavelet_noise_experiments"
cd "$BASE_DIR" || exit 1

# Run for 252C_19bar
cd 252/BOS_12_11_4
echo "Running in 252C_19bar"
mpiexec -n 4 SU2_CFD config_pr_0_0.cfg
mpiexec -n 4 SU2_CFD config_pr_0.cfg
mpiexec -n 4 SU2_CFD config_pr.cfg
mpiexec -n 4 SU2_CFD config_pr_1.cfg
cd "$BASE_DIR"

# Run for 252C_9bar
cd 252/BOS_12_11_7
echo "Running in 252C_9bar"
mpiexec -n 4 SU2_CFD config_pr.cfg
mpiexec -n 4 SU2_CFD config_pr_1.cfg
cd "$BASE_DIR"

echo "All simulations completed."
