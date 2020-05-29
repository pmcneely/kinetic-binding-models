#!/usr/bin/env python3

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import equations

# data_location = 'data/single_ligand/FA_Data_All.xlsx'
# df = pd.read_excel(data_location, header=None)

# tspan = np.array(df.iloc[:, 0])
# converted_receptor_complexes = np.array(df.iloc[:, 2])
# converted_receptor_complexes -= np.mean(converted_receptor_complexes[0:4])

simulation_location = 'data/simulations/single_ligand.json'
with open(simulation_location, 'r') as fd:
    data = json.load(fd)
fd.close()

converted_receptor_complexes = np.array(data['labeled_receptor_complexes'])
tspan = np.array(data['time'])

units = 1e9  # 1e9 for nM, 1e6 for μM, etc

L1 = 30e-9
R = 800e-9
m = np.array([R, L1, 0]) * units
alpha = 0.06

optimized_result = equations.one_ligand_fixed_alpha(tspan, converted_receptor_complexes, m, alpha)
bmin = optimized_result.x
print(f"Corresponding Kd: {bmin[1]/bmin[0]:.3f}")

Kd = 22  # nM
fixed_Kd_optimization = equations.one_ligand_fixed_kd(tspan, converted_receptor_complexes, m, 22)
params = fixed_Kd_optimization.x
print(f"k_on: {params[0]*Kd:.3e} k_off: {params[0]:.3e} Corresponding Kd: {params[0]*Kd/params[0]:.3f}")
print(f"Computed α value: {params[1]*100:.1f}%")



# print(solution.y)

# z = solution.sol(t)

# plt.semilogy(t/3600, z.T)
# plt.xlabel('t')
# plt.legend(['R', 'L', 'RL'])

# plt.savefig('round_two.png')
