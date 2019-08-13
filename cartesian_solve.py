#%% import
import ldope
import numpy as np
from numpy import sqrt, deg2rad
from scipy.integrate import solve_ivp

#%% 初始条件
CASE_1 = ((-6598.210e3, 1163.440e3, 0, 1.160e3, 6.578e3, 3.857e3),
          (6062.180e3, 3500e3, 0, -3.268e3, 5.660e3, 3.773e3))

CASE_2 = ((536.291e3, -6008.420e3, -3470.030e3, 7.402e3, 1.063e3, -0.138e3),
          (-5824.010e3, 1571.490e3, -3470.030e3, 2.332e3, 7.105e3, -0.138e3))

CASE_3 = ((-6598.210e3, 1163.440e3, 0, -0.458e3, -2.598e3, 7.248e3),
          (-6755e3, -3900e3, 0, 3.520e3, -6.097e3, 1.241e3))

CASES = (CASE_1, CASE_2, CASE_3)


def cartesian_solve(case_index, lb, ub, norm_process=True):
    state_p0 = CASES[case_index - 1][0]
    state_e0 = CASES[case_index - 1][1]

    opt_individual, opt_value, time_cost = ldope.cartesian_solve_fcn(
        state_p0, state_e0, lb, ub, norm_process=norm_process)

    print(opt_individual)
    print(opt_value)
    print(time_cost)

    costate_p0, costate_e0, tf_norm = ldope.cartesian_individual_convert_fcn(
        opt_individual)

    step = 0.05
    t_span = 0, tf_norm
    t_eval = np.arange(0, tf_norm, step)
    t_eval = np.append(t_eval, tf_norm)

    if norm_process:
        file_name = './data/cartesian/{}_norm.data'.format(case_index)
    else:
        file_name = './data/cartesian/{}.data'.format(case_index)

    with open(file_name, 'a') as data:
        data.write('{} '.format(opt_value))
        data.write('{} '.format(time_cost))
        for ele in opt_individual:
            data.write('{} '.format(ele))
        data.write('\n')


#%% 求解
if __name__ == "__main__":

    case_index = 3

    lb = -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, 1
    ub = 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 5

    cartesian_solve(case_index, lb, ub, True)
    cartesian_solve(case_index, lb, ub, False)
