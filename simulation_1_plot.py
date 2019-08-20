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
    # 获取最优值，计算时间和最优个体（归一化）
    cartesian_opt_value, cartesian_cost_time, cartesian_individual_norm = \
        plot_cartesian_result.find_best_cartesian_record(case_index, True)
    spherical_opt_value, spherical_cost_time, spherical_individual_norm = \
        plot_spherical_result.find_best_spherical_record(case_index, True)

    # 打印最优个体，保留四位小数
    print('best cartesian individual of case{}:'.format(case_index))
    print(float('%.4f' % cartesian_opt_value))
    print(float('%.4f' % cartesian_cost_time))
    print(', \, '.join(['%.4f' % ele for ele in cartesian_individual_norm]))
    print('best spherical individual of case{}:'.format(case_index))
    print(float('%.4f' % spherical_opt_value))
    print(float('%.4f' % spherical_cost_time))
    print(', \, '.join(['%.4f' % ele for ele in spherical_individual_norm]))

    # 绘制3D坐标轴
    fig_norm, ax_norm = plot_util.plot_3D_frame(
        'case {} norm frame'.format(case_index), True)

    # 绘制起始点
    plot_spherical_result.plot_spherical_initial_point(
        case_index,
        ax_norm,
        True,
        p_label='Initial Position P{}'.format(case_index),
        e_label='Initial Position E{}'.format(case_index))

    # 绘制轨迹
    plot_cartesian_result.plot_cartesian_trajectory(
        case_index,
        cartesian_individual_norm,
        ax_norm,
        True,
        p_label='Trajectory P{} in CM'.format(case_index),
        e_label='Trajectory E{} in CM'.format(case_index))
    plot_spherical_result.plot_spherical_trajectory(
        case_index,
        spherical_individual_norm,
        ax_norm,
        True,
        p_color=seaborn.xkcd_rgb['purple'],
        e_color=seaborn.xkcd_rgb['blue'],
        p_label='Trajectory P{} in SM'.format(case_index),
        e_label='Trajectory E{} in SM'.format(case_index),
        p_linestyle='-.',
        e_linestyle=':')

    # 绘制控制变量坐标轴
    fig_control, (ax_control_alpha_p, ax_control_alpha_e, ax_control_beta_p,
                  ax_control_beta_e) = plot_util.plot_control_frame(
                      'case {} control frame'.format(case_index))

    # 绘制控制变量
    plot_cartesian_result.plot_cartesian_control(
        case_index, cartesian_individual_norm, ax_control_alpha_p,
        ax_control_alpha_e, ax_control_beta_p, ax_control_beta_e)
    plot_spherical_result.plot_spherical_control(
        case_index,
        spherical_individual_norm,
        ax_control_alpha_p,
        ax_control_alpha_e,
        ax_control_beta_p,
        ax_control_beta_e,
        p_color=seaborn.xkcd_rgb['purple'],
        e_color=seaborn.xkcd_rgb['blue'],
        p_linestyle='-.',
        e_linestyle=':')

    # 绘制图例
    ax_norm.legend(loc='upper right')

    if case_index == 1:
        offset = 0.3
    else:
        offset = 0.5
    x_locator = plt.MultipleLocator(offset)

    xlim = (ax_control_alpha_p.get_xlim())
    new_xlim = [xlim[0], xlim[1] + offset]
    ax_control_alpha_p.xaxis.set_major_locator(x_locator)
    ax_control_alpha_e.xaxis.set_major_locator(x_locator)
    ax_control_alpha_p.set_xlim(new_xlim)
    ax_control_alpha_e.set_xlim(new_xlim)
    ax_control_alpha_p.legend(loc='upper right')
    ax_control_alpha_e.legend(bbox_to_anchor=[1, 0.65])

    xlim = (ax_control_beta_p.get_xlim())
    new_xlim = [xlim[0], xlim[1] + offset]
    ax_control_beta_p.xaxis.set_major_locator(x_locator)
    ax_control_beta_e.xaxis.set_major_locator(x_locator)
    ax_control_beta_p.set_xlim(new_xlim)
    ax_control_beta_e.set_xlim(new_xlim)
    ax_control_beta_p.legend(loc='upper right')
    ax_control_beta_e.legend(bbox_to_anchor=[1, 0.61])

    # 调整图比例
    fig_norm.tight_layout()
    fig_control.tight_layout()

# 显示
plt.show()
