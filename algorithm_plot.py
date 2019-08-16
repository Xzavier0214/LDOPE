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

cartesian_index = 4
spherical_index = 7

# 获取最优值，计算时间和最优个体（归一化）
cartesian_opt_value, cartesian_cost_time, cartesian_individual_norm = \
    plot_cartesian_result.find_best_cartesian_record(cartesian_index, True)
spherical_opt_value, spherical_cost_time, spherical_individual_norm = \
    plot_spherical_result.find_best_spherical_record(spherical_index, True)

# 绘制3D坐标轴
fig_c, ax_c = plot_util.plot_3D_frame('cartesian trajectory', True)
fig_s, ax_s = plot_util.plot_3D_frame('spherical trajectory', True)

# 绘制起始点
plot_cartesian_result.plot_cartesian_initial_point(cartesian_index, ax_c, True)
plot_spherical_result.plot_spherical_initial_point(spherical_index, ax_s, True)

# 绘制轨迹
plot_cartesian_result.plot_cartesian_trajectory(cartesian_index,
                                                cartesian_individual_norm,
                                                ax_c, True)
plot_spherical_result.plot_spherical_trajectory(spherical_index,
                                                spherical_individual_norm,
                                                ax_s, True)

# 绘制控制变量坐标轴
fig_control_c, (
    ax_control_alpha_p_c, ax_control_alpha_e_c, ax_control_beta_p_c,
    ax_control_beta_e_c) = plot_util.plot_control_frame('cartesian control')
fig_control_s, (
    ax_control_alpha_p_s, ax_control_alpha_e_s, ax_control_beta_p_s,
    ax_control_beta_e_s) = plot_util.plot_control_frame('spherical control')

# 绘制控制变量
plot_cartesian_result.plot_cartesian_control(
    cartesian_index, cartesian_individual_norm, ax_control_alpha_p_c,
    ax_control_alpha_e_c, ax_control_beta_p_c, ax_control_beta_e_c)
plot_spherical_result.plot_spherical_control(
    spherical_index, spherical_individual_norm, ax_control_alpha_p_s,
    ax_control_alpha_e_s, ax_control_beta_p_s, ax_control_beta_e_s)

fig_c.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
fig_s.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
fig_control_c.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
fig_control_s.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)

# 调整图比例
fig_c.tight_layout()
fig_s.tight_layout()
fig_control_c.tight_layout()
fig_control_s.tight_layout()

# 显示
plt.show()
