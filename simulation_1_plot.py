#%% import
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy import sin, cos, sqrt
import seaborn
from scipy.integrate import solve_ivp
import ldope
import plot_spherical_result
import plot_cartesian_result
import plot_util

for case_index in range(1, 4):
    spherical_opt_value, spherical_cost_time, spherical_individual_norm = \
        plot_spherical_result.find_best_spherical_record(case_index, True)
    _, _, spherical_individual = \
        plot_spherical_result.find_best_spherical_record(case_index, False)
    cartesian_opt_value, cartesian_cost_time, cartesian_individual_norm = \
        plot_cartesian_result.find_best_cartesian_record(case_index, True)
    _, _, cartesian_individual = \
        plot_cartesian_result.find_best_cartesian_record(case_index, False)

    fig_norm = plot_util.plot_norm_frame(
        'case {} norm frame'.format(case_index))
    fig = plot_util.plot_frame('case {} frame'.format(case_index))

    plot_spherical_result.plot_spherical_result_by_individual(
        case_index, spherical_individual_norm, fig_norm, True)
    plot_cartesian_result.plot_cartesian_result_by_individual(
        case_index,
        cartesian_individual_norm,
        fig_norm,
        True,
        p_color=seaborn.xkcd_rgb['purple'],
        e_color=seaborn.xkcd_rgb['blue'])

    plot_spherical_result.plot_spherical_result_by_individual(
        case_index, spherical_individual, fig, False)
    plot_cartesian_result.plot_cartesian_result_by_individual(
        case_index,
        cartesian_individual,
        fig,
        False,
        p_color=seaborn.xkcd_rgb['purple'],
        e_color=seaborn.xkcd_rgb['blue'])

plt.show()
