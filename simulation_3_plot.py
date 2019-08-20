import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy import sin, cos, sqrt, deg2rad
import seaborn
from scipy.integrate import solve_ivp
import ldope
import plot_spherical_result
import plot_cartesian_result
import plot_util

spherical_opt_value_4, spherical_cost_time_4, spherical_individual_norm_4 = \
    plot_spherical_result.find_best_spherical_record(4, True)
spherical_opt_value_5, spherical_cost_time_5, spherical_individual_norm_5 = \
    plot_spherical_result.find_best_spherical_record(5, True)
spherical_opt_value_6, spherical_cost_time_6, spherical_individual_norm_6 = \
    plot_spherical_result.find_best_spherical_record(6, True)

# 绘制3D坐标轴
fig_p, ax_p = plot_util.plot_3D_frame('pursuer longitude', True)
fig_e, ax_e = plot_util.plot_3D_frame('evader longitude', True)

# 绘制起始点
plot_spherical_result.plot_spherical_initial_point(
    4,
    ax_p,
    True,
    p_color=seaborn.xkcd_rgb['red'],
    p_label='Initial Position P4',
    side='p')
plot_spherical_result.plot_spherical_initial_point(
    5,
    ax_p,
    True,
    p_color=seaborn.xkcd_rgb['green'],
    p_label='Initial Position P5',
    side='p')
plot_spherical_result.plot_spherical_initial_point(
    6,
    ax_p,
    True,
    p_color=seaborn.xkcd_rgb['blue'],
    p_label='Initial Position P6',
    side='p')

plot_spherical_result.plot_spherical_initial_point(
    4,
    ax_e,
    True,
    e_color=seaborn.xkcd_rgb['red'],
    e_label='Initial Position E4',
    side='e')
plot_spherical_result.plot_spherical_initial_point(
    5,
    ax_e,
    True,
    e_color=seaborn.xkcd_rgb['green'],
    e_label='Initial Position E5',
    side='e')
plot_spherical_result.plot_spherical_initial_point(
    6,
    ax_e,
    True,
    e_color=seaborn.xkcd_rgb['blue'],
    e_label='Initial Position E6',
    side='e')

# 绘制轨迹
plot_spherical_result.plot_spherical_trajectory(
    4,
    spherical_individual_norm_4,
    ax_p,
    True,
    p_color=seaborn.xkcd_rgb['red'],
    p_label='Trajectory P4',
    p_linestyle='-',
    side='p')
plot_spherical_result.plot_spherical_trajectory(
    5,
    spherical_individual_norm_5,
    ax_p,
    True,
    p_color=seaborn.xkcd_rgb['green'],
    p_label='Trajectory P5',
    p_linestyle='--',
    side='p')
plot_spherical_result.plot_spherical_trajectory(
    6,
    spherical_individual_norm_6,
    ax_p,
    True,
    p_color=seaborn.xkcd_rgb['blue'],
    p_label='Trajectory P6',
    p_linestyle='-.',
    side='p')

plot_spherical_result.plot_spherical_trajectory(
    4,
    spherical_individual_norm_4,
    ax_e,
    True,
    e_color=seaborn.xkcd_rgb['red'],
    e_label='Trajectory E4',
    e_linestyle='-',
    side='e')
plot_spherical_result.plot_spherical_trajectory(
    5,
    spherical_individual_norm_5,
    ax_e,
    True,
    e_color=seaborn.xkcd_rgb['green'],
    e_label='Trajectory E5',
    e_linestyle='--',
    side='e')
plot_spherical_result.plot_spherical_trajectory(
    6,
    spherical_individual_norm_6,
    ax_e,
    True,
    e_color=seaborn.xkcd_rgb['blue'],
    e_label='Trajectory E6',
    e_linestyle='-.',
    side='e')

# 绘制时间-半径坐标轴
fig_tr, (ax_tr_p, ax_tr_e) = plot_util.plot_tr_frame(r't vs. $\bar{r}')

# 绘制时间-半径
plot_spherical_result.plot_spherical_tr(4,
                                        spherical_individual_norm_4,
                                        ax_tr_p,
                                        ax_tr_e,
                                        p_color=seaborn.xkcd_rgb['red'],
                                        e_color=seaborn.xkcd_rgb['red'],
                                        p_label=r'P4 $\bar{r}$',
                                        e_label=r'E4 $\bar{r}$',
                                        p_linestyle='-',
                                        e_linestyle='-')
plot_spherical_result.plot_spherical_tr(5,
                                        spherical_individual_norm_5,
                                        ax_tr_p,
                                        ax_tr_e,
                                        p_color=seaborn.xkcd_rgb['green'],
                                        e_color=seaborn.xkcd_rgb['green'],
                                        p_label=r'P5 $\bar{r}$',
                                        e_label=r'E5 $\bar{r}$',
                                        p_linestyle='--',
                                        e_linestyle='--')
plot_spherical_result.plot_spherical_tr(6,
                                        spherical_individual_norm_6,
                                        ax_tr_p,
                                        ax_tr_e,
                                        p_color=seaborn.xkcd_rgb['blue'],
                                        e_color=seaborn.xkcd_rgb['blue'],
                                        p_label=r'P6 $\bar{r}$',
                                        e_label=r'E6 $\bar{r}$',
                                        p_linestyle='-.',
                                        e_linestyle='-.')

# 绘制时间-纬度坐标轴
fig_tphi, (ax_tphi_p,
           ax_tphi_e) = plot_util.plot_tphi_frame(r't vs. ${\varphi}$')

# 绘制时间-纬度
plot_spherical_result.plot_spherical_tphi(4,
                                          spherical_individual_norm_4,
                                          ax_tphi_p,
                                          ax_tphi_e,
                                          p_color=seaborn.xkcd_rgb['red'],
                                          e_color=seaborn.xkcd_rgb['red'],
                                          p_label=r'P4 $\varphi$',
                                          e_label=r'E4 $\varphi$',
                                          p_linestyle='-',
                                          e_linestyle='-')
plot_spherical_result.plot_spherical_tphi(5,
                                          spherical_individual_norm_5,
                                          ax_tphi_p,
                                          ax_tphi_e,
                                          p_color=seaborn.xkcd_rgb['green'],
                                          e_color=seaborn.xkcd_rgb['green'],
                                          p_label=r'P5 $\varphi$',
                                          e_label=r'E5 $\varphi$',
                                          p_linestyle='--',
                                          e_linestyle='--')
plot_spherical_result.plot_spherical_tphi(6,
                                          spherical_individual_norm_6,
                                          ax_tphi_p,
                                          ax_tphi_e,
                                          p_color=seaborn.xkcd_rgb['blue'],
                                          e_color=seaborn.xkcd_rgb['blue'],
                                          p_label=r'P6 $\varphi$',
                                          e_label=r'E6 $\varphi$',
                                          p_linestyle='-.',
                                          e_linestyle='-.')

# 绘制时间-经度坐标轴
fig_txi, (ax_txi_p, ax_txi_e) = plot_util.plot_txi_frame(r't vs. ${\xi}$')

# 绘制时间-经度
plot_spherical_result.plot_spherical_txi(4,
                                         spherical_individual_norm_4,
                                         ax_txi_p,
                                         ax_txi_e,
                                         p_color=seaborn.xkcd_rgb['red'],
                                         e_color=seaborn.xkcd_rgb['red'],
                                         p_label=r'P4 $\xi$',
                                         e_label=r'E4 $\xi$',
                                         p_linestyle='-',
                                         e_linestyle='-')
plot_spherical_result.plot_spherical_txi(5,
                                         spherical_individual_norm_5,
                                         ax_txi_p,
                                         ax_txi_e,
                                         p_color=seaborn.xkcd_rgb['green'],
                                         e_color=seaborn.xkcd_rgb['green'],
                                         p_label=r'P5 $\xi$',
                                         e_label=r'E5 $\xi$',
                                         p_linestyle='--',
                                         e_linestyle='--')
plot_spherical_result.plot_spherical_txi(6,
                                         spherical_individual_norm_6,
                                         ax_txi_p,
                                         ax_txi_e,
                                         p_color=seaborn.xkcd_rgb['blue'],
                                         e_color=seaborn.xkcd_rgb['blue'],
                                         p_label=r'P6 $\xi$',
                                         e_label=r'E6 $\xi$',
                                         p_linestyle='-.',
                                         e_linestyle='-.')

# 绘制时间-经度坐标轴（modified）
fig_txi_modified, (ax_txi_p_modified,
                   ax_txi_e_modified) = plot_util.plot_txi_frame(
                       r't vs. ${\hat{\xi}}$', True)

# 绘制时间-经度（modified）
plot_spherical_result.plot_spherical_txi(4,
                                         spherical_individual_norm_4,
                                         ax_txi_p_modified,
                                         ax_txi_e_modified,
                                         p_color=seaborn.xkcd_rgb['red'],
                                         e_color=seaborn.xkcd_rgb['red'],
                                         p_label=r'P4 $\hat{\xi}$',
                                         e_label=r'E4 $\hat{\xi}$',
                                         p_linestyle='-',
                                         e_linestyle='-',
                                         p_base_xi=0,
                                         e_base_xi=0)
plot_spherical_result.plot_spherical_txi(5,
                                         spherical_individual_norm_5,
                                         ax_txi_p_modified,
                                         ax_txi_e_modified,
                                         p_color=seaborn.xkcd_rgb['green'],
                                         e_color=seaborn.xkcd_rgb['green'],
                                         p_label=r'P5 $\hat{\xi}$',
                                         e_label=r'E5 $\hat{\xi}$',
                                         p_linestyle='--',
                                         e_linestyle='--',
                                         p_base_xi=deg2rad(20),
                                         e_base_xi=deg2rad(20))
plot_spherical_result.plot_spherical_txi(6,
                                         spherical_individual_norm_6,
                                         ax_txi_p_modified,
                                         ax_txi_e_modified,
                                         p_color=seaborn.xkcd_rgb['blue'],
                                         e_color=seaborn.xkcd_rgb['blue'],
                                         p_label=r'P6 $\hat{\xi}$',
                                         e_label=r'E6 $\hat{\xi}$',
                                         p_linestyle='-.',
                                         e_linestyle='-.',
                                         p_base_xi=deg2rad(40),
                                         e_base_xi=deg2rad(40))

# 绘制3D坐标轴
fig_same, ax_same = plot_util.plot_3D_frame('same control', True)

# 绘制起始点
plot_spherical_result.plot_spherical_initial_point(
    5,
    ax_same,
    True,
    p_color=seaborn.xkcd_rgb['red'],
    p_label='Initial Position P5',
    e_color=seaborn.xkcd_rgb['green'],
    e_label='Initial Position E5')

# 绘制相同控制
plot_spherical_result.plot_same_control(5, spherical_individual_norm_5, 6,
                                        spherical_individual_norm_6, ax_same,
                                        p_o_label='P5 Origin',
                                        e_o_label='E5 Origin',
                                        p_label='P5 in P6 Controller',
                                        e_label='E5 in E6 Controller')

# 显示图例
ax_p.legend(loc='upper right')
ax_e.legend(loc='upper right')
ax_tr_p.legend()
ax_tphi_p.legend()
ax_txi_p.legend()
ax_txi_p_modified.legend()
ax_tr_e.legend()
ax_tphi_e.legend()
ax_txi_e.legend()
ax_txi_e_modified.legend()
ax_same.legend(loc='upper right')

# 调整图比例
fig_p.tight_layout()
fig_e.tight_layout()
fig_tr.tight_layout()
fig_tphi.tight_layout()
fig_txi.tight_layout()
fig_txi_modified.tight_layout()
fig_same.tight_layout()

# 显示
plt.show()
