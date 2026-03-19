This folder contains the CFD simulation for the ORCHID nozzle at a compressibility factor of ~0.75

config_pr.cfg runs the simulation with a 1st order spatial accuracy.
Flow.vtu is the resulting solution.

In order to enable a spatial accuracy of the 2nd order, one final restart is required.
config_pr_1.cfg use the solution from config_pr.cfg to increase the spatial accuracy.
flow_MUSCL.vtu is the final solution.

The equation of state chosen for the simulations is Peng-Robinson.
The file BOS_12_11_1_CFD.dat contains the extracted data at the nozzle midplane from the file flow_MUSCL.vtu.

The throat height of the simulated nozzle is 8.609 mm
