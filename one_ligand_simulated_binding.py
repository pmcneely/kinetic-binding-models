#!/usr/bin/env python3

import json
import numpy as np
import matplotlib.pyplot as plt

import equations

data_output = 'data/simulations/single_ligand.json'

tspan = np.array([0, 120 * 60])  # 2 hour window

units = 1e9  # 1e9 for nM, 1e6 for μM, etc

L1 = 30e-9
R = 800e-9
alpha = 0.06
m = np.array([R, L1, 0]) * units

k = np.array([1e-5, 2.2e-4])

print(f"Simulated conditions: α: {alpha * 100:.1f}% Kd: {k[1]/k[0]:.3f}nM")
simulated_binding = equations.simulate_one_ligand_one_receptor_binding(k, m, tspan, alpha)

result = {
    'time': simulated_binding.t.tolist(),
    'available_receptor': simulated_binding.y[0,:].tolist(),
    'labeled_ligand': simulated_binding.y[1,:].tolist(),
    'labeled_receptor_complexes': simulated_binding.y[2,:].tolist(),
    'label_kinetic_parameters': k.tolist(),
    'initial_conditions': m.tolist(),
    'units': 'nM',
    'alpha': alpha
    }

with open(data_output, 'w') as fd:
    fd.write(json.dumps(result))
fd.close()
