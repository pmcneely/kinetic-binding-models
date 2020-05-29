import copy
import numpy as np
import scipy.integrate as integrate
import scipy.optimize as optimize


def one_ligand_one_receptor_fixed_alpha_system(t, m, k):
    val = np.array((
        - (k[0]*m[0]*m[1] - k[1]*m[2]),
        - (k[0]*m[0]*m[1] - k[1]*m[2]),
        k[0]*m[0]*m[1] - k[1]*m[2]
    ))
    return val


def one_ligand_one_receptor_fixed_kd_system(t, m, k, Kd):
    val = np.array((
        - (k[0]*m[0]*m[1] - k[0]*Kd*m[2]),
        - (k[0]*m[0]*m[1] - k[0]*Kd*m[2]),
        k[0]*m[0]*m[1] - k[0]*Kd*m[2]
    ))
    return val


def two_ligand_one_receptor_fixed_alpha_system(t, m, k, fa_k):
    val = np.array((
        - (k[0]*m[0]*m[1] - k[1]*m[2]),  # Change in unlabeled ligand (L2)
        # change in total RL1 and RL2 complexes (loss of R)
        - (k[0]*m[0]*m[1] - k[1]*m[2]) - (fa_k[0]*m[0]*m[3] - fa_k[1]*m[4]),
          (k[0]*m[0]*m[1] - k[1]*m[2]),  # Change in RL2 complexes
        - (fa_k[0]*m[0]*m[3] - fa_k[1]*m[4]),  # Change in L1 concentration
          (fa_k[0]*m[0]*m[3] - fa_k[1]*m[4])  # Change in RL1 complexes
    ))
    return val


def two_ligand_one_receptor_fixed_ki_system(t, m, k, fa_k, Ki):
    val = np.array((
        - (k[0]*m[0]*m[1] - k[0]*Ki*m[2]),  # Change in unlabeled ligand (L2)
        # change in total RL1 and RL2 complexes (loss of R)
        - (k[0]*m[0]*m[1] - k[0]*Ki*m[2]) - (fa_k[0]*m[0]*m[3] - fa_k[1]*m[4]),
          (k[0]*m[0]*m[1] - k[0]*Ki*m[2]),  # Change in RL2 complexes
        - (fa_k[0]*m[0]*m[3] - fa_k[1]*m[4]),  # Change in L1 concentration
          (fa_k[0]*m[0]*m[3] - fa_k[1]*m[4])  # Change in RL1 complexes
    ))
    return val

def simulate_one_ligand_one_receptor_binding(k, initial_conditions, tspan, alpha):

    time_points = np.linspace(*tspan, 101)
    m = copy.deepcopy(initial_conditions)
    m[0] *= alpha

    return integrate.solve_ivp(
        one_ligand_one_receptor_fixed_alpha_system,
        tspan, m, args=(k,),
        t_eval=time_points, method='BDF', dense_output=True
    )

def simulate_two_ligand_one_receptor_binding(k, initial_conditions, fa_k, tspan, alpha):

    time_points = np.linspace(*tspan, 100)
    m = copy.deepcopy(initial_conditions)
    m[0] *= alpha

    return integrate.solve_ivp(
        two_ligand_one_receptor_fixed_alpha_system,
        tspan, m, args=(k, fa_k),
        t_eval=time_points, method='BDF', dense_output=True
    )

def one_ligand_fixed_alpha_err_estimate(k, initial_conditions, tspan, time_points, data):

    solution = integrate.solve_ivp(
        one_ligand_one_receptor_fixed_alpha_system,
        tspan, initial_conditions, args=(k,),
        t_eval=time_points, method='BDF', dense_output=True
    )

    # Plotting here?

    return np.array((solution.y[2, :] - data))


def one_ligand_fixed_kd_err_estimate(k, initial_conditions, R0, Kd, tspan, time_points, data):

    initial_conditions[0] = R0 * k[1]

    solution = integrate.solve_ivp(
        one_ligand_one_receptor_fixed_kd_system,
        tspan, initial_conditions, args=(k, Kd),
        t_eval=time_points, method='BDF', dense_output=True
    )

    # Plotting here?

    return np.array((solution.y[2, :] - data))


def two_ligand_fixed_alpha_err_estimate(k, initial_conditions, fa_k, tspan, time_points, data):

    solution = integrate.solve_ivp(
        two_ligand_one_receptor_fixed_alpha_system, 
        tspan, initial_conditions, args=(k, fa_k),
        t_eval=time_points, method='BDF', dense_output=True
    )

    # Plotting here?

    return np.array((solution.y[4, :] - data))


def two_ligand_fixed_ki_err_estimate(k, R0, Kd, initial_conditions, fa_k, tspan, time_points, data):

    initial_conditions[0] = R0 * k[1]

    solution = integrate.solve_ivp(
        two_ligand_one_receptor_fixed_ki_system, tspan, initial_conditions, args=(
            k, fa_k, Kd),
        t_eval=time_points, method='BDF', dense_output=True
    )

    # Plotting here?

    return np.array((solution.y[4, :] - data))


def one_ligand_fixed_alpha(time, data, initial_conditions, alpha):

    xtol = 1e-10
    max_evaluations = 2.5e3
    lower_bound = np.array([0, 0])
    upper_bound = np.array([np.inf, np.inf])
    bounds = (lower_bound, upper_bound)

    tspan = np.array([time[0], time[-1]])
    k = np.array([1e-9, 1e-3])
    m = copy.deepcopy(initial_conditions)

    assert(alpha >= 0 and alpha <= 1), \
        "α must be in [0, 1.0] (ie, % of total receptor concentration)"
    m[0] *= alpha

    optimized_params = optimize.least_squares(
        one_ligand_fixed_alpha_err_estimate,
        k, xtol=xtol, max_nfev=max_evaluations, bounds=bounds,
        args=(m, tspan, time, data)
    )

    return optimized_params


def one_ligand_fixed_kd(time, data, initial_conditions, Kd):

    xtol = 1e-10
    max_evaluations = 2.5e3
    lower_bound = np.array([0, 0])
    upper_bound = np.array([np.inf, 1.0])
    bounds = (lower_bound, upper_bound)

    tspan = np.array([time[0], time[-1]])
    k = np.array([1e-9, 0.05])
    m = copy.deepcopy(initial_conditions)
    R0 = m[0]

    optimized_params = optimize.least_squares(
        one_ligand_fixed_kd_err_estimate,
        k, xtol=xtol, max_nfev=max_evaluations, bounds=bounds,
        args=(m, R0, Kd, tspan, time, data)
    )

    return optimized_params


def two_ligand_fixed_alpha(time, data, initial_conditions, params, alpha):

    xtol = 1e-10
    max_evaluations = 2.5e3
    lower_bound = np.array([0, 0])
    upper_bound = np.array([np.inf, np.inf])
    bounds = (lower_bound, upper_bound)

    tspan = np.array([time[0], time[-1]])
    k = np.array([1e-9, 1e-2])
    m = copy.deepcopy(initial_conditions)

    assert(alpha >= 0 and alpha <= 1), \
        "α must be in [0, 1.0] (ie, % of total receptor concentration)"
    m[0] *= alpha

    optimized_params = optimize.least_squares(
        two_ligand_fixed_alpha_err_estimate,
        k, xtol=xtol, max_nfev=max_evaluations, bounds=bounds,
        args=(m, params, tspan, time, data)
    )

    return optimized_params


def two_ligand_fixed_ki(time, data, initial_conditions, params, Ki):

    xtol = 1e-10
    max_evaluations = 2.5e3
    lower_bound = np.array([0, 0])
    upper_bound = np.array([np.inf, np.inf])
    bounds = (lower_bound, upper_bound)

    tspan = np.array([time[0], time[-1]])
    k = np.array([1e-9, 0.2])  # Initial guess for alpha is k[1](%)
    m = copy.deepcopy(initial_conditions)
    R0 = m[0]

    optimized_params = optimize.least_squares(
        two_ligand_fixed_ki_err_estimate,
        k, xtol=xtol, max_nfev=max_evaluations, bounds=bounds,
        args=(R0, Ki, m, params, tspan, time, data)
    )

    return optimized_params
