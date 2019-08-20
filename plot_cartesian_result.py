#%% import
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy import sin, cos, sqrt
import seaborn
from scipy.integrate import solve_ivp
import ldope
import solve_cartesian_model
import plot_util
import pandas


# 搜索最优个体记录
def find_best_cartesian_record(case_index, norm=True):
    # 笛卡尔模型归一化与常规结果是分开保存的
    if norm:
        file_name = './data/cartesian/{}_norm.data'.format(case_index)
    else:
        file_name = './data/cartesian/{}.data'.format(case_index)

    # 读取记录
    data = pandas.read_csv(file_name,
                           ' ',
                           header=None,
                           usecols=range(ldope.CARTESIAN_INDIVIDUAL_SIZE + 2))
    # 根据optValue取最小记录
    best_record = data.nsmallest(1, 0).values[0]

    # 返回 最优值，计算时间，最优记录
    return float(best_record[0]), float(best_record[1]), tuple(
        best_record[2:ldope.CARTESIAN_INDIVIDUAL_SIZE + 2])


# 绘制起始点
def plot_cartesian_initial_point(case_index,
                                 ax,
                                 norm=True,
                                 p_color=seaborn.xkcd_rgb['red'],
                                 e_color=seaborn.xkcd_rgb['green'],
                                 p_label='Initial Position P',
                                 e_label='Initial Position E'):
    # 初始状态
    state_p0 = solve_cartesian_model.CASES[case_index - 1][0]
    state_e0 = solve_cartesian_model.CASES[case_index - 1][1]
    if norm:
        state_p0 = ldope.cartesian_state_norm_fcn(state_p0)
        state_e0 = ldope.cartesian_state_norm_fcn(state_e0)

    # 初始点
    xp = state_p0[0]
    yp = state_p0[1]
    zp = state_p0[2]
    xe = state_e0[0]
    ye = state_e0[1]
    ze = state_e0[2]

    # 绘制
    ax.scatter(xp, yp, zp, color=p_color, marker='*', s=50, label=p_label)
    ax.scatter(xe, ye, ze, color=e_color, marker='s', s=25, label=e_label)


# 根据个体绘制轨道结果
def plot_cartesian_trajectory(case_index,
                              individual,
                              ax,
                              norm=True,
                              p_color=seaborn.xkcd_rgb['red'],
                              e_color=seaborn.xkcd_rgb['green'],
                              p_linestyle='-',
                              e_linestyle='--',
                              p_label='Trajectory P in CM',
                              e_label='Trajectory E in CM'):
    # 初始状态量
    state_p0 = solve_cartesian_model.CASES[case_index - 1][0]
    state_e0 = solve_cartesian_model.CASES[case_index - 1][1]
    if norm:
        state_p0 = ldope.cartesian_state_norm_fcn(state_p0)
        state_e0 = ldope.cartesian_state_norm_fcn(state_e0)

    # 初始协态量
    costate_p0, costate_e0, tf_norm = ldope.cartesian_individual_convert_fcn(
        individual)

    # 积分时段
    if norm:
        step = 0.05
        t_span = 0, tf_norm
        t_eval = np.arange(0, tf_norm, step)
        t_eval = np.append(t_eval, tf_norm)
    else:
        step = 0.05 * ldope.TU
        t_span = 0, tf_norm * ldope.TU
        t_eval = np.arange(0, tf_norm * ldope.TU, step)
        t_eval = np.append(t_eval, tf_norm * ldope.TU)

    # 积分
    result = solve_ivp(
        lambda t, y: ldope.cartesian_ext_state_fcn(tuple(y.tolist()), norm),
        t_span=np.array(t_span),
        y0=np.array(state_p0 + state_e0 + costate_p0 + costate_e0),
        t_eval=t_eval)

    # 绘制向量计算
    xp, yp, zp = [], [], []
    xe, ye, ze = [], [], []
    for each_state in result.y.T:
        xp.append(each_state[0])
        yp.append(each_state[1])
        zp.append(each_state[2])
        xe.append(each_state[6])
        ye.append(each_state[7])
        ze.append(each_state[8])

    # 绘制
    ax.plot(xp, yp, zp, color=p_color, linestyle=p_linestyle, label=p_label)
    ax.plot(xe, ye, ze, color=e_color, linestyle=e_linestyle, label=e_label)


# 根据个体绘制控制变量
def plot_cartesian_control(case_index,
                           individual,
                           ax_control_alpha_p,
                           ax_control_alpha_e,
                           ax_control_beta_p,
                           ax_control_beta_e,
                           p_color=seaborn.xkcd_rgb['red'],
                           e_color=seaborn.xkcd_rgb['green'],
                           p_linestyle='-',
                           e_linestyle='--'):
    # 初始状态量
    state_p0 = solve_cartesian_model.CASES[case_index - 1][0]
    state_e0 = solve_cartesian_model.CASES[case_index - 1][1]
    state_p0 = ldope.cartesian_state_norm_fcn(state_p0)
    state_e0 = ldope.cartesian_state_norm_fcn(state_e0)

    # 初始协态量
    costate_p0, costate_e0, tf_norm = ldope.cartesian_individual_convert_fcn(
        individual)

    # 积分时段
    step = 0.05
    t_span = 0, tf_norm
    t_eval = np.arange(0, tf_norm, step)
    t_eval = np.append(t_eval, tf_norm)

    # 积分
    result = solve_ivp(
        lambda t, y: ldope.cartesian_ext_state_fcn(tuple(y.tolist()), True),
        t_span=np.array(t_span),
        y0=np.array(state_p0 + state_e0 + costate_p0 + costate_e0),
        t_eval=t_eval)

    # 绘制向量计算
    t_line = []
    alpha_p, alpha_e = [], []
    beta_p, beta_e = [], []
    for i, t in enumerate(result.t):
        each_state = result.y.T[i]
        t_line.append(t)
        control_p = ldope.cartesian_control_fcn(each_state[0:6],
                                                each_state[12:18], 'p')
        alpha_p.append(control_p[0])
        beta_p.append(control_p[1])

        control_e = ldope.cartesian_control_fcn(each_state[6:12],
                                                each_state[18:24], 'e')
        alpha_e.append(control_e[0])
        beta_e.append(control_e[1])

    # 绘制
    ax_control_alpha_p.plot(t_line,
                            alpha_p,
                            color=p_color,
                            linestyle=p_linestyle,
                            label=r'$\alpha_P$')
    ax_control_alpha_p.set_ylabel(ax_control_alpha_p.get_ylabel() + ' ' +
                                  r'$\alpha_P$')

    ax_control_alpha_e.plot(t_line,
                            alpha_e,
                            color=e_color,
                            linestyle=e_linestyle,
                            label=r'$\alpha_E$')
    ax_control_alpha_e.set_ylabel(ax_control_alpha_e.get_ylabel() + ' ' +
                                  r'$\alpha_E$')

    ax_control_beta_p.plot(t_line,
                           beta_p,
                           color=p_color,
                           linestyle=p_linestyle,
                           label=r'$\beta_P$')
    ax_control_beta_p.set_ylabel(ax_control_beta_p.get_ylabel() + ' ' +
                                 r'$\beta_P$')

    ax_control_beta_e.plot(t_line,
                           beta_e,
                           color=e_color,
                           linestyle=e_linestyle,
                           label=r'$\beta_E$')
    ax_control_beta_e.set_ylabel(ax_control_beta_e.get_ylabel() + ' ' +
                                 r'$\beta_E$')


# 绘制时间-距离
def plot_cartesian_td(case_index,
                      individual,
                      ax,
                      norm=True,
                      color=seaborn.xkcd_rgb['red'],
                      linestyle='-',
                      label='t vs. D'):
    # 初始状态量
    state_p0 = solve_cartesian_model.CASES[case_index - 1][0]
    state_e0 = solve_cartesian_model.CASES[case_index - 1][1]
    if norm:
        state_p0 = ldope.cartesian_state_norm_fcn(state_p0)
        state_e0 = ldope.cartesian_state_norm_fcn(state_e0)

    # 初始协态量
    costate_p0, costate_e0, tf_norm = ldope.cartesian_individual_convert_fcn(
        individual)

    # 积分时段
    if norm:
        step = 0.05
        t_span = 0, tf_norm
        t_eval = np.arange(0, tf_norm, step)
        t_eval = np.append(t_eval, tf_norm)
    else:
        step = 0.05 * ldope.TU
        t_span = 0, tf_norm * ldope.TU
        t_eval = np.arange(0, tf_norm * ldope.TU, step)
        t_eval = np.append(t_eval, tf_norm * ldope.TU)

    # 积分
    result = solve_ivp(
        lambda t, y: ldope.cartesian_ext_state_fcn(tuple(y.tolist()), norm),
        t_span=np.array(t_span),
        y0=np.array(state_p0 + state_e0 + costate_p0 + costate_e0),
        t_eval=t_eval)

    # 绘制向量计算
    r = []
    t_line = []
    for i, t in enumerate(result.t):
        if norm:
            t_line.append(t)
        else:
            t_line.append(t / ldope.TU)
        each_state = result.y.T[i]
        xp = each_state[0]
        yp = each_state[1]
        zp = each_state[2]
        xe = each_state[6]
        ye = each_state[7]
        ze = each_state[8]
        r.append(sqrt((xp - xe)**2 + (yp - ye)**2 + (zp - ze)**2))

    # 绘制
    ax.plot(t_line, r, color=color, linestyle=linestyle, label=label)

    ax.scatter(t_line[-1], r[-1], color=color, marker='*')
    # ax.text(t_line[-3],
    #         r[-3],
    #         '(' + ('%.2f' % t_line[-1]) + ', ' + ('%.2f' % r[-1]) + ')',
    #         fontsize=10)


if __name__ == "__main__":
    # 算例指定和获取最优个体
    case_index = 1
    _, _, individual_norm = find_best_cartesian_record(case_index, True)
    _, _, individual = find_best_cartesian_record(case_index, False)

    # 绘制3D坐标轴
    fig_norm, ax_norm = plot_util.plot_3D_frame(
        'case {} norm frame'.format(case_index), True)
    fig, ax = plot_util.plot_3D_frame('case {} frame'.format(case_index),
                                      False)

    # 绘制起始点
    plot_cartesian_initial_point(case_index, ax_norm, True)
    plot_cartesian_initial_point(case_index, ax, False)

    # 绘制轨迹
    plot_cartesian_trajectory(case_index, individual_norm, ax_norm, True)
    plot_cartesian_trajectory(case_index, individual, ax, False)

    # 绘制控制变量坐标系
    fig_control, (ax_control_alpha, ax2_alpha, ax_control_beta,
                  ax2_beta) = plot_util.plot_control_frame(
                      'case {} control frame'.format(case_index))

    # 绘制控制变量
    plot_cartesian_control(case_index, individual_norm, ax_control_alpha,
                           ax2_alpha, ax_control_beta, ax2_beta)

    # 绘制时间-距离坐标轴
    fig_tr_norm, ax_tr_norm = plot_util.plot_td_frame(
        'case {} tr norm frame'.format(case_index), True)
    fig_tr, ax_tr = plot_util.plot_td_frame(
        'case {} tr frame'.format(case_index), False)

    # 绘制时间-距离图
    plot_cartesian_td(case_index, individual_norm, ax_tr_norm, True)
    plot_cartesian_td(case_index, individual, ax_tr, False)

    # 显示图例
    ax_norm.legend()
    ax.legend()
    ax_control_alpha.legend()
    ax_control_beta.legend()
    ax_tr_norm.legend()
    ax_tr.legend()

    # 调整图比例
    fig_norm.tight_layout()
    fig.tight_layout()
    fig_control.tight_layout()
    fig_tr_norm.tight_layout()
    fig_tr.tight_layout()

    plt.show()
