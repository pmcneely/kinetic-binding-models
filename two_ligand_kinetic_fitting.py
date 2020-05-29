#!/usr/bin/env python3

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import equations

# data_location = 'data/two_ligands/Ado65nM_Indiv_Curves.xlsx'
# df = pd.read_excel(data_location, header=None)

# tspan = np.round(np.array(df.iloc[:, 0]) * 60)
# converted_receptor_complexes = np.array(df.iloc[:, 2])
# converted_receptor_complexes -= np.mean(converted_receptor_complexes[0:4])

simulation_location = 'data/simulations/two_ligand.json'
with open(simulation_location, 'r') as fd:
    data = json.load(fd)
fd.close()

converted_receptor_complexes = np.array(data['labeled_receptor_complexes'])
tspan = np.array(data['time'])
units = 1e9  # 1e9 for nM, 1e6 for μM, etc

L1 = 30e-9
L2 = 65e-9
R = 800e-9
m = np.array([R, L2, 0, L1, 0]) * units
print(f"Using initial conditions {m}")

# fitc_apec_params = np.array([1.62e-5, 3.55e-4])
fitc_apec_params = np.array([1e-5, 2.2e-4])
print(f"FITC-APEC effective Kd: {fitc_apec_params[1]/fitc_apec_params[0]:.3f}nM")

alphas = np.arange(0.5,20.6,0.5) / 100
for alpha in alphas:
    optimized_result = equations.two_ligand_fixed_alpha(tspan, converted_receptor_complexes, m, fitc_apec_params, alpha)
    bmin = optimized_result.x
    print(f"α: {alpha:.3f} - Corresponding Ki: {bmin[1]/bmin[0]:.3f}")

Ki = 12.5  # nM
fixed_Ki_optimization = equations.two_ligand_fixed_ki(tspan, converted_receptor_complexes, m, fitc_apec_params, Ki)
params = fixed_Ki_optimization.x
print(f"k_on: {params[0]*Ki:.3e} k_off: {params[0]:.3e} Corresponding Kd: {params[0]*Ki/params[0]:.3f}")
print(f"Computed α value: {params[1]*100:.1f}%")

# print(solution.y)

# z = solution.sol(t)

# plt.semilogy(t/3600, z.T)
# plt.xlabel('t')
# plt.legend(['R', 'L', 'RL'])

# plt.savefig('round_two.png')
