#%% import
import ldope
import numpy as np
from numpy import sqrt, deg2rad
from scipy.integrate import solve_ivp

#%% 初始条件
case_index = 3

case_1 = ((6700e3, 7.713e3, 0, deg2rad(170), deg2rad(0), deg2rad(-210)),
          (7000e3, 7.546e3, 0, deg2rad(30), deg2rad(0), deg2rad(30)))

case_2 = ((7037.658e3, 7.438e3, deg2rad(-2.1), deg2rad(-84.9), deg2rad(-29.9),
           deg2rad(-2.4)), (7037.658e3, 7.438e3, deg2rad(-2.1), deg2rad(164.9),
                            deg2rad(-29.9), deg2rad(-177.6)))

case_3 = ((6700e3, 7.713e3, deg2rad(0), deg2rad(170), deg2rad(0), deg2rad(70)),
          (7800e3, 7.149e3, deg2rad(0), deg2rad(210), deg2rad(0), deg2rad(10)))

cases = (case_1, case_2, case_3)

state_p0 = cases[case_index - 1][0]
state_e0 = cases[case_index - 1][1]

lb = -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, 1
ub = 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 5

#%% 求解
if __name__ == "__main__":
    opt_individual, opt_value, time_cost = ldope.spherical_solve_fcn(
        state_p0, state_e0, lb, ub)

    print(opt_individual)
    print(opt_value)
    print(time_cost)

    #%% 存储
    costate_p0, costate_e0, tf_norm = ldope.spherical_individual_convert_fcn(
        opt_individual)

    step = 0.05
    t_span = 0, tf_norm
    t_eval = np.arange(0, tf_norm, step)
    t_eval = np.append(t_eval, tf_norm)

    file_name = './data/spherical/{}.data'.format(case_index)

    with open(file_name, 'a') as data:
        for ele in opt_individual:
            data.write('{} '.format(ele))
        data.write('{} '.format(opt_value))
        data.write('{}\n'.format(time_cost))
