import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy import sin, cos, sqrt, pi
import seaborn
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import ldope
import solve_spherical_model
import plot_util
import pandas


# 搜索最优个体记录
def find_best_spherical_record(case_index, norm=True):
    # 读取记录
    file_name = './data/spherical/{}.data'.format(case_index)
    data = pandas.read_csv(file_name,
                           ' ',
                           header=None,
                           usecols=range(ldope.SPHERICAL_INDIVIDUAL_SIZE + 2))

    # 根据optValue取最小记录
    best_record = data.nsmallest(1, 0).values[0]

    # 返回 最优值，计算时间，最优记录
    return float(best_record[0]), float(best_record[1]), tuple(
        best_record[2:ldope.SPHERICAL_INDIVIDUAL_SIZE + 2])


# 绘制起始点
def plot_spherical_initial_point(case_index,
                                 ax,
                                 norm=True,
                                 p_color=seaborn.xkcd_rgb['red'],
                                 e_color=seaborn.xkcd_rgb['green'],
                                 p_label='Initial Position P',
                                 e_label='Initial Position E',
                                 side='both'):
    # 初始状态
    state_p0 = solve_spherical_model.CASES[case_index - 1][0]
    state_e0 = solve_spherical_model.CASES[case_index - 1][1]
    if norm:
        state_p0 = ldope.spherical_state_norm_fcn(state_p0)
        state_e0 = ldope.spherical_state_norm_fcn(state_e0)

    # 初始点
    r_p, r_e = state_p0[0], state_e0[0]
    xi_p, xi_e = state_p0[3], state_e0[3]
    phi_p, phi_e = state_p0[4], state_e0[4]
    xp = (r_p * cos(phi_p) * cos(xi_p))
    yp = (r_p * cos(phi_p) * sin(xi_p))
    zp = (r_p * sin(phi_p))
    xe = (r_e * cos(phi_e) * cos(xi_e))
    ye = (r_e * cos(phi_e) * sin(xi_e))
    ze = (r_e * sin(phi_e))

    # 绘制
    if side == 'both' or side == 'p':
        ax.scatter(xp, yp, zp, color=p_color, marker='*', s=50, label=p_label)
    if side == 'both' or side == 'e':
        ax.scatter(xe, ye, ze, color=e_color, marker='s', s=25, label=e_label)


# 根据个体绘制轨道结果
def plot_spherical_trajectory(case_index,
                              individual,
                              ax,
                              norm=True,
                              p_color=seaborn.xkcd_rgb['red'],
                              e_color=seaborn.xkcd_rgb['green'],
                              p_linestyle='-',
                              e_linestyle='--',
                              p_label='Trajectory P in SM',
                              e_label='Trajectory E in SM',
                              side='both'):
    # 初始状态量
    state_p0 = solve_spherical_model.CASES[case_index - 1][0]
    state_e0 = solve_spherical_model.CASES[case_index - 1][1]
    if norm:
        state_p0 = ldope.spherical_state_norm_fcn(state_p0)
        state_e0 = ldope.spherical_state_norm_fcn(state_e0)

    # 初始协态量
    costate_p0, costate_e0, tf_norm = ldope.spherical_individual_convert_fcn(
        individual)

    # 积分时段
    if (norm):
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
        lambda t, y: ldope.spherical_ext_state_fcn(tuple(y.tolist()), norm),
        t_span=np.array(t_span),
        y0=np.array(state_p0 + state_e0 + costate_p0 + costate_e0),
        t_eval=t_eval)

    # 绘制向量计算
    xp, yp, zp = [], [], []
    xe, ye, ze = [], [], []
    for each_state in result.y.T:
        r_p, r_e = each_state[0], each_state[6]
        xi_p, xi_e = each_state[3], each_state[9]
        phi_p, phi_e = each_state[4], each_state[10]
        xp.append(r_p * cos(phi_p) * cos(xi_p))
        yp.append(r_p * cos(phi_p) * sin(xi_p))
        zp.append(r_p * sin(phi_p))
        xe.append(r_e * cos(phi_e) * cos(xi_e))
        ye.append(r_e * cos(phi_e) * sin(xi_e))
        ze.append(r_e * sin(phi_e))

    # 绘制
    if side == 'both' or side == 'p':
        ax.plot(xp,
                yp,
                zp,
                color=p_color,
                linestyle=p_linestyle,
                label=p_label)
    if side == 'both' or side == 'e':
        ax.plot(xe,
                ye,
                ze,
                color=e_color,
                linestyle=e_linestyle,
                label=e_label)


# 根据个体绘制控制变量
def plot_spherical_control(case_index,
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
    state_p0 = solve_spherical_model.CASES[case_index - 1][0]
    state_e0 = solve_spherical_model.CASES[case_index - 1][1]
    state_p0 = ldope.spherical_state_norm_fcn(state_p0)
    state_e0 = ldope.spherical_state_norm_fcn(state_e0)

    # 初始协态量
    costate_p0, costate_e0, tf_norm = ldope.spherical_individual_convert_fcn(
        individual)

    # 积分时段
    step = 0.05
    t_span = 0, tf_norm
    t_eval = np.arange(0, tf_norm, step)
    t_eval = np.append(t_eval, tf_norm)

    # 积分
    result = solve_ivp(
        lambda t, y: ldope.spherical_ext_state_fcn(tuple(y.tolist()), True),
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
        control_p = ldope.spherical_control_fcn(each_state[0:6],
                                                each_state[12:18], 'p')
        alpha_p.append(control_p[0])
        beta_p.append(control_p[1])

        control_e = ldope.spherical_control_fcn(each_state[6:12],
                                                each_state[18:24], 'e')
        alpha_e.append(control_e[0])
        beta_e.append(control_e[1])

    # 绘制
    ax_control_alpha_p.plot(t_line,
                            alpha_p,
                            color=p_color,
                            linestyle=p_linestyle,
                            label=r'$\hat{\alpha}_P$')
    ax_control_alpha_p.set_ylabel(ax_control_alpha_p.get_ylabel() + ' ' +
                                  r'$\hat{\alpha}_P$')

    ax_control_alpha_e.plot(t_line,
                            alpha_e,
                            color=e_color,
                            linestyle=e_linestyle,
                            label=r'$\hat{\alpha}_E$')
    ax_control_alpha_e.set_ylabel(ax_control_alpha_e.get_ylabel() + ' ' +
                                  r'$\hat{\alpha}_E$')

    ax_control_beta_p.plot(t_line,
                           beta_p,
                           color=p_color,
                           linestyle=p_linestyle,
                           label=r'$\hat{\beta}_P$')
    ax_control_beta_p.set_ylabel(ax_control_beta_p.get_ylabel() + ' ' +
                                 r'$\hat{\beta}_P$')

    ax_control_beta_e.plot(t_line,
                           beta_e,
                           color=e_color,
                           linestyle=e_linestyle,
                           label=r'$\hat{\beta}_E$')
    ax_control_beta_e.set_ylabel(ax_control_beta_e.get_ylabel() + ' ' +
                                 r'$\hat{\beta}_E$')


# 绘制时间-距离
def plot_spherical_td(case_index,
                      individual,
                      ax,
                      norm=True,
                      color=seaborn.xkcd_rgb['red'],
                      linestyle='-',
                      label='t vs. D'):
    # 初始状态量
    state_p0 = solve_spherical_model.CASES[case_index - 1][0]
    state_e0 = solve_spherical_model.CASES[case_index - 1][1]
    if norm:
        state_p0 = ldope.spherical_state_norm_fcn(state_p0)
        state_e0 = ldope.spherical_state_norm_fcn(state_e0)

    # 初始协态量
    costate_p0, costate_e0, tf_norm = ldope.spherical_individual_convert_fcn(
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
        lambda t, y: ldope.spherical_ext_state_fcn(tuple(y.tolist()), norm),
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
        r_p, r_e = each_state[0], each_state[6]
        xi_p, xi_e = each_state[3], each_state[9]
        phi_p, phi_e = each_state[4], each_state[10]
        xp = r_p * cos(phi_p) * cos(xi_p)
        yp = r_p * cos(phi_p) * sin(xi_p)
        zp = r_p * sin(phi_p)
        xe = r_e * cos(phi_e) * cos(xi_e)
        ye = r_e * cos(phi_e) * sin(xi_e)
        ze = r_e * sin(phi_e)
        r.append(sqrt((xp - xe)**2 + (yp - ye)**2 + (zp - ze)**2))

    # 绘制
    ax.plot(t_line, r, color=color, linestyle=linestyle, label=label)

    ax.scatter(t_line[-1], r[-1], color=color, marker='*')
    # ax.text(t_line[-3],
    #         r[-3],
    #         '(' + ('%.2f' % t_line[-1]) + ', ' + ('%.2f' % r[-1]) + ')',
    #         fontsize=10)


# 绘制时间-半径图
def plot_spherical_tr(case_index,
                      individual,
                      ax_p,
                      ax_e,
                      p_color=seaborn.xkcd_rgb['red'],
                      p_linestyle='-',
                      p_label='Pursuer t vs. r',
                      e_color=seaborn.xkcd_rgb['green'],
                      e_linestyle='--',
                      e_label='Evader t vs. r'):
    # 初始状态量
    state_p0 = solve_spherical_model.CASES[case_index - 1][0]
    state_e0 = solve_spherical_model.CASES[case_index - 1][1]
    state_p0 = ldope.spherical_state_norm_fcn(state_p0)
    state_e0 = ldope.spherical_state_norm_fcn(state_e0)

    # 初始协态量
    costate_p0, costate_e0, tf_norm = ldope.spherical_individual_convert_fcn(
        individual)

    # 积分时段
    step = 0.05
    t_span = 0, tf_norm
    t_eval = np.arange(0, tf_norm, step)
    t_eval = np.append(t_eval, tf_norm)

    # 积分
    result = solve_ivp(
        lambda t, y: ldope.spherical_ext_state_fcn(tuple(y.tolist()), True),
        t_span=np.array(t_span),
        y0=np.array(state_p0 + state_e0 + costate_p0 + costate_e0),
        t_eval=t_eval)

    # 绘制向量计算
    t_line = []
    r_p_line = []
    r_e_line = []
    for i, t in enumerate(result.t):
        t_line.append(t)
        each_state = result.y.T[i]
        r_p, r_e = each_state[0], each_state[6]
        xi_p, xi_e = each_state[3], each_state[9]
        phi_p, phi_e = each_state[4], each_state[10]
        xp = r_p * cos(phi_p) * cos(xi_p)
        yp = r_p * cos(phi_p) * sin(xi_p)
        zp = r_p * sin(phi_p)
        xe = r_e * cos(phi_e) * cos(xi_e)
        ye = r_e * cos(phi_e) * sin(xi_e)
        ze = r_e * sin(phi_e)

        r_p_line.append(r_p)
        r_e_line.append(r_e)

    # 绘制
    ax_p.semilogx(t_line,
                  r_p_line,
                  basex=2,
                  color=p_color,
                  linestyle=p_linestyle,
                  label=p_label)
    ax_e.semilogx(t_line,
                  r_e_line,
                  basex=2,
                  color=e_color,
                  linestyle=e_linestyle,
                  label=e_label)


# 绘制时间-纬度图
def plot_spherical_tphi(case_index,
                        individual,
                        ax_p,
                        ax_e,
                        p_color=seaborn.xkcd_rgb['red'],
                        p_linestyle='-',
                        p_label=r'Pursuer t vs. $\varphi$',
                        e_color=seaborn.xkcd_rgb['green'],
                        e_linestyle='--',
                        e_label=r'Evader t vs. $\varphi$'):
    # 初始状态量
    state_p0 = solve_spherical_model.CASES[case_index - 1][0]
    state_e0 = solve_spherical_model.CASES[case_index - 1][1]
    state_p0 = ldope.spherical_state_norm_fcn(state_p0)
    state_e0 = ldope.spherical_state_norm_fcn(state_e0)

    # 初始协态量
    costate_p0, costate_e0, tf_norm = ldope.spherical_individual_convert_fcn(
        individual)

    # 积分时段
    step = 0.05
    t_span = 0, tf_norm
    t_eval = np.arange(0, tf_norm, step)
    t_eval = np.append(t_eval, tf_norm)

    # 积分
    result = solve_ivp(
        lambda t, y: ldope.spherical_ext_state_fcn(tuple(y.tolist()), True),
        t_span=np.array(t_span),
        y0=np.array(state_p0 + state_e0 + costate_p0 + costate_e0),
        t_eval=t_eval)

    # 绘制向量计算
    t_line = []
    phi_p_line = []
    phi_e_line = []
    for i, t in enumerate(result.t):
        t_line.append(t)
        each_state = result.y.T[i]
        r_p, r_e = each_state[0], each_state[6]
        xi_p, xi_e = each_state[3], each_state[9]
        phi_p, phi_e = each_state[4], each_state[10]
        xp = r_p * cos(phi_p) * cos(xi_p)
        yp = r_p * cos(phi_p) * sin(xi_p)
        zp = r_p * sin(phi_p)
        xe = r_e * cos(phi_e) * cos(xi_e)
        ye = r_e * cos(phi_e) * sin(xi_e)
        ze = r_e * sin(phi_e)

        phi_p_line.append(phi_p)
        phi_e_line.append(phi_e)

    # 绘制
    ax_p.semilogx(t_line,
                  phi_p_line,
                  basex=2,
                  color=p_color,
                  linestyle=p_linestyle,
                  label=p_label)
    ax_e.semilogx(t_line,
                  phi_e_line,
                  basex=2,
                  color=e_color,
                  linestyle=e_linestyle,
                  label=e_label)


# 绘制时间经度图
def plot_spherical_txi(case_index,
                       individual,
                       ax_p,
                       ax_e,
                       p_color=seaborn.xkcd_rgb['red'],
                       p_linestyle='-',
                       p_label=r'Pursuer t vs. $\xi$',
                       e_color=seaborn.xkcd_rgb['green'],
                       e_linestyle='--',
                       e_label=r'Evader t vs. $\xi$',
                       p_base_xi=0,
                       e_base_xi=0):
    # 初始状态量
    state_p0 = solve_spherical_model.CASES[case_index - 1][0]
    state_e0 = solve_spherical_model.CASES[case_index - 1][1]
    state_p0 = ldope.spherical_state_norm_fcn(state_p0)
    state_e0 = ldope.spherical_state_norm_fcn(state_e0)

    # 初始协态量
    costate_p0, costate_e0, tf_norm = ldope.spherical_individual_convert_fcn(
        individual)

    # 积分时段
    step = 0.05
    t_span = 0, tf_norm
    t_eval = np.arange(0, tf_norm, step)
    t_eval = np.append(t_eval, tf_norm)

    # 积分
    result = solve_ivp(
        lambda t, y: ldope.spherical_ext_state_fcn(tuple(y.tolist()), True),
        t_span=np.array(t_span),
        y0=np.array(state_p0 + state_e0 + costate_p0 + costate_e0),
        t_eval=t_eval)

    # 绘制向量计算
    t_line = []
    xi_p_line = []
    xi_e_line = []
    for i, t in enumerate(result.t):
        t_line.append(t)
        each_state = result.y.T[i]
        r_p, r_e = each_state[0], each_state[6]
        xi_p, xi_e = each_state[3], each_state[9]
        phi_p, phi_e = each_state[4], each_state[10]
        xp = r_p * cos(phi_p) * cos(xi_p)
        yp = r_p * cos(phi_p) * sin(xi_p)
        zp = r_p * sin(phi_p)
        xe = r_e * cos(phi_e) * cos(xi_e)
        ye = r_e * cos(phi_e) * sin(xi_e)
        ze = r_e * sin(phi_e)

        xi_p_line.append(xi_p)
        xi_e_line.append(xi_e)

    # 绘制
    ax_p.semilogx(t_line,
                  np.array(xi_p_line[:]) - p_base_xi,
                  basex=2,
                  color=p_color,
                  linestyle=p_linestyle,
                  label=p_label)
    ax_e.semilogx(t_line,
                  np.array(xi_p_line[:]) - e_base_xi,
                  basex=2,
                  color=e_color,
                  linestyle=e_linestyle,
                  label=e_label)


# 绘制采用相同控制方式图
def plot_same_control(case_index,
                      individual,
                      control_case_index,
                      control_individual,
                      ax,
                      p_o_color=seaborn.xkcd_rgb['red'],
                      p_o_linestyle='-',
                      p_o_label='P origin',
                      e_o_color=seaborn.xkcd_rgb['green'],
                      e_o_linestyle='--',
                      e_o_label='E origin',
                      p_color=seaborn.xkcd_rgb['purple'],
                      p_linestyle='-.',
                      p_label='P',
                      e_color=seaborn.xkcd_rgb['blue'],
                      e_linestyle=':',
                      e_label='E'):
    # 控制初始状态量
    state_control_p0 = solve_spherical_model.CASES[control_case_index - 1][0]
    state_control_e0 = solve_spherical_model.CASES[control_case_index - 1][1]
    state_control_p0 = ldope.spherical_state_norm_fcn(state_control_p0)
    state_control_e0 = ldope.spherical_state_norm_fcn(state_control_e0)

    # 控制初始协态量
    costate_control_p0, costate_control_e0, tf_control_norm = \
        ldope.spherical_individual_convert_fcn(control_individual)

    # 控制积分时段
    step_control = 0.05
    t_span_control = 0, tf_control_norm
    t_eval_control = np.arange(0, tf_control_norm, step_control)
    t_eval_control = np.append(t_eval_control, tf_control_norm)

    # 积分
    result_control = solve_ivp(
        lambda t, y: ldope.spherical_ext_state_fcn(tuple(y.tolist()), True),
        t_span=np.array(t_span_control),
        y0=np.array(state_control_p0 + state_control_e0 + costate_control_p0 +
                    costate_control_e0),
        t_eval=t_eval_control)

    # 控制向量计算
    t_line = []
    alpha_p, alpha_e = [], []
    beta_p, beta_e = [], []
    for i, t in enumerate(result_control.t):
        t_line.append(t)
        each_state = result_control.y.T[i]
        control_p = ldope.spherical_control_fcn(each_state[0:6],
                                                each_state[12:18], 'p')
        alpha_p.append(control_p[0] if control_p[0] > -2 else control_p[0] +
                       2 * pi)
        beta_p.append(control_p[1] if control_p[1] > 1 else control_p[1] +
                      2 * pi)

        control_e = ldope.spherical_control_fcn(each_state[6:12],
                                                each_state[18:24], 'e')
        alpha_e.append(control_e[0])
        beta_e.append(control_e[1])

    alpha_p_fcn = interp1d(t_line, alpha_p, kind='cubic')
    beta_p_fcn = interp1d(t_line, beta_p, kind='cubic')

    alpha_e_fcn = interp1d(t_line, alpha_e, kind='cubic')
    beta_e_fcn = interp1d(t_line, beta_e, kind='cubic')

    def state_p_fcn(t, y: np.ndarray):
        return ldope.spherical_state_fcn(tuple(y.tolist()),
                                         (alpha_p_fcn(t), beta_p_fcn(t)),
                                         side='p',
                                         norm=True)

    def state_e_fcn(t, y: np.ndarray):
        return ldope.spherical_state_fcn(tuple(y.tolist()),
                                         (alpha_e_fcn(t), beta_e_fcn(t)),
                                         side='e',
                                         norm=True)

    # 初始状态量
    state_p0 = solve_spherical_model.CASES[case_index - 1][0]
    state_e0 = solve_spherical_model.CASES[case_index - 1][1]
    state_p0 = ldope.spherical_state_norm_fcn(state_p0)
    state_e0 = ldope.spherical_state_norm_fcn(state_e0)

    # 初始协态量
    costate_p0, costate_e0, tf_norm = ldope.spherical_individual_convert_fcn(
        individual)

    # 积分时段
    step = 0.05
    t_span = 0, tf_norm
    t_eval = np.arange(0, tf_norm, step)
    t_eval = np.append(t_eval, tf_norm)

    print(tf_control_norm)
    print(tf_norm)

    # 积分
    result_o = solve_ivp(
        lambda t, y: ldope.spherical_ext_state_fcn(tuple(y.tolist()), True),
        t_span=np.array(t_span),
        y0=np.array(state_p0 + state_e0 + costate_p0 + costate_e0),
        t_eval=t_eval)
    result_p = solve_ivp(state_p_fcn,
                         t_span=np.array(t_span),
                         y0=np.array(state_p0),
                         t_eval=t_eval)
    result_e = solve_ivp(state_e_fcn,
                         t_span=np.array(t_span),
                         y0=np.array(state_e0),
                         t_eval=t_eval)

    # 绘制向量计算
    xp_o, yp_o, zp_o = [], [], []
    xe_o, ye_o, ze_o = [], [], []
    for each_state in result_o.y.T:
        r_p, r_e = each_state[0], each_state[6]
        xi_p, xi_e = each_state[3], each_state[9]
        phi_p, phi_e = each_state[4], each_state[10]
        xp_o.append(r_p * cos(phi_p) * cos(xi_p))
        yp_o.append(r_p * cos(phi_p) * sin(xi_p))
        zp_o.append(r_p * sin(phi_p))
        xe_o.append(r_e * cos(phi_e) * cos(xi_e))
        ye_o.append(r_e * cos(phi_e) * sin(xi_e))
        ze_o.append(r_e * sin(phi_e))

    xp, yp, zp = [], [], []
    for each_state in result_p.y.T:
        r_p = each_state[0]
        xi_p = each_state[3]
        phi_p = each_state[4]
        xp.append(r_p * cos(phi_p) * cos(xi_p))
        yp.append(r_p * cos(phi_p) * sin(xi_p))
        zp.append(r_p * sin(phi_p))

    xe, ye, ze = [], [], []
    for each_state in result_e.y.T:
        r_e = each_state[0]
        xi_e = each_state[3]
        phi_e = each_state[4]
        xe.append(r_e * cos(phi_e) * cos(xi_e))
        ye.append(r_e * cos(phi_e) * sin(xi_e))
        ze.append(r_e * sin(phi_e))

    # 绘制
    ax.plot(xp_o,
            yp_o,
            zp_o,
            color=p_o_color,
            linestyle=p_o_linestyle,
            label=p_o_label)
    ax.plot(xe_o,
            ye_o,
            ze_o,
            color=e_o_color,
            linestyle=e_o_linestyle,
            label=e_o_label)
    ax.plot(xp, yp, zp, color=p_color, linestyle=p_linestyle, label=p_label)
    ax.plot(xe_o,
            ye_o,
            ze_o,
            color=e_color,
            linestyle=e_linestyle,
            label=e_label)


if __name__ == "__main__":
    # 算例指定和获取最优个体
    case_index = 1
    _, _, individual_norm = find_best_spherical_record(case_index, True)
    _, _, individual = find_best_spherical_record(case_index, False)

    # 绘制3D坐标轴
    fig_norm, ax_norm = plot_util.plot_3D_frame(
        'case {} norm frame'.format(case_index), True)
    fig, ax = plot_util.plot_3D_frame('case {} frame'.format(case_index),
                                      False)

    # 绘制起始点
    plot_spherical_initial_point(case_index, ax_norm, True)
    plot_spherical_initial_point(case_index, ax, False)

    # 绘制轨迹
    plot_spherical_trajectory(case_index, individual_norm, ax_norm, True)
    plot_spherical_trajectory(case_index, individual, ax, False)

    # 绘制控制变量坐标系
    fig_control, (ax_control_alpha, ax2_alpha, ax_control_beta,
                  ax2_beta) = plot_util.plot_control_frame(
                      'case {} control frame'.format(case_index))

    # 绘制控制变量
    plot_spherical_control(case_index, individual_norm, ax_control_alpha,
                           ax2_alpha, ax_control_beta, ax2_beta)

    # 绘制时间-距离坐标轴
    fig_tr_norm, ax_tr_norm = plot_util.plot_td_frame(
        'case {} tr norm frame'.format(case_index), True)
    fig_tr, ax_tr = plot_util.plot_td_frame(
        'case {} tr frame'.format(case_index), False)

    # 绘制时间-距离图
    plot_spherical_td(case_index, individual_norm, ax_tr_norm, True)
    plot_spherical_td(case_index, individual, ax_tr, False)

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
