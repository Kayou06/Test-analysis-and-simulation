This folder contains the CFD simulation for the ORCHID nozzle at a supercritical condition.

The thermodynamic model cannot reach convergency instantaneously; for this reason multiple restarts are required.

config_pr_0_0.cfg is the first configuration file to be run. (same temperature, but ~15 bar)
config_pr_0.cfg is the second configuration file to be run. (same temperature, but ~17bar)
config_pr.cfg is the third configuration file to be run, in this file the final thermodynamic conditions are reached.
Flow.vtu is the resulting solution.

In order to enable a spatial accuracy of the 2nd order, one final restart is required.
config_pr_1.cfg use the solution from config_pr.cfg to increase the spatial accuracy.
flow_MUSCL.vtu is the final solution.

The equation of state chosen for the simulations is Peng-Robinson.
The file BOS_12_11_4_CFD.dat contains the extracted data at the nozzle midplane from the file flow_MUSCL.vtu.

The throat height of the simulated nozzle is 8.664 mm.
