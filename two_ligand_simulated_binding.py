#!/usr/bin/env python3

import json
import numpy as np
import matplotlib.pyplot as plt

import equations

data_output = 'data/simulations/two_ligand.json'

tspan = np.array([0, 120 * 60])  # 2 hour window

units = 1e9  # 1e9 for nM, 1e6 for μM, etc

L1 = 30e-9
L2 = 65e-9
R = 800e-9
alpha = 0.08
m = np.array([R, L2, 0, L1, 0]) * units

fa_k = np.array([1e-5, 2.2e-4])
k = np.array([2.4e-5, 3.0e-4])

print(f"Simulated conditions: α: {alpha * 100:.1f}% Kd: {k[1]/k[0]:.3f}nM")
simulated_binding = equations.simulate_two_ligand_one_receptor_binding(k, m, fa_k, tspan, alpha)

result = {
    'time': simulated_binding.t.tolist(),
    'available_receptor': simulated_binding.y[0,:].tolist(),
    'unlabeled_ligand': simulated_binding.y[1,:].tolist(),
    'unlabeled_receptor_complexes': simulated_binding.y[2,:].tolist(),
    'labeled_ligand': simulated_binding.y[3,:].tolist(),
    'labeled_receptor_complexes': simulated_binding.y[4,:].tolist(),
    'label_kinetic_parameters': fa_k.tolist(),
    'unlabelled_kinetic_parameters': k.tolist(),
    'initial_conditions': m.tolist(),
    'units': 'nM',
    'alpha': alpha
    }

with open(data_output, 'w') as fd:
    fd.write(json.dumps(result))
fd.close()
