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

case_index = 3

# 获取最优值，计算时间和最优个体
cartesian_opt_value, cartesian_cost_time, cartesian_individual_norm = \
    plot_cartesian_result.find_best_cartesian_record(case_index, True)
_, _, cartesian_individual = \
    plot_cartesian_result.find_best_cartesian_record(case_index, False)
spherical_opt_value, spherical_cost_time, spherical_individual_norm = \
    plot_spherical_result.find_best_spherical_record(case_index, True)
_, _, spherical_individual = \
    plot_spherical_result.find_best_spherical_record(case_index, False)

# 绘制3D坐标轴
fig, ax = plot_util.plot_3D_frame('case {} frame'.format(case_index), False)

# 绘制起始点
plot_spherical_result.plot_spherical_initial_point(case_index, ax, False)

# 绘制轨迹
plot_cartesian_result.plot_cartesian_trajectory(case_index,
                                                cartesian_individual, ax,
                                                False)
plot_spherical_result.plot_spherical_trajectory(
    case_index,
    spherical_individual,
    ax,
    False,
    p_color=seaborn.xkcd_rgb['purple'],
    e_color=seaborn.xkcd_rgb['blue'],
    p_linestyle='-.',
    e_linestyle=':')

# 绘制时间-距离坐标轴
fig_td_norm, ax_td_norm = plot_util.plot_td_frame(
    'case {} tr norm frame'.format(case_index), True)
fig_td, ax_td = plot_util.plot_td_frame('case {} tr frame'.format(case_index),
                                        False)

# 绘制时间-距离图
plot_cartesian_result.plot_cartesian_td(case_index, cartesian_individual_norm,
                                        ax_td_norm, True)
plot_cartesian_result.plot_cartesian_td(case_index, cartesian_individual,
                                        ax_td, False)
plot_spherical_result.plot_spherical_td(case_index,
                                        spherical_individual_norm,
                                        ax_td_norm,
                                        True,
                                        color=seaborn.xkcd_rgb['purple'],
                                        linestyle='-.')
plot_spherical_result.plot_spherical_td(case_index,
                                        spherical_individual,
                                        ax_td,
                                        False,
                                        color=seaborn.xkcd_rgb['purple'],
                                        linestyle='-.')

# 调整图比例
fig.tight_layout()
fig_td_norm.tight_layout()
fig_td.tight_layout()

# 显示
plt.show()
