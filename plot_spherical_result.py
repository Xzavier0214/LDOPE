import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy import sin, cos, sqrt
import seaborn
from scipy.integrate import solve_ivp
import ldope
import solve_spherical_model
import plot_util
import pandas


def find_best_spherical_record(case_index, norm=True):
    file_name = './data/spherical/{}.data'.format(case_index)
    data = pandas.read_csv(file_name,
                           ' ',
                           header=None,
                           usecols=range(ldope.SPHERICAL_INDIVIDUAL_SIZE + 2))
    best_record = data.nsmallest(1, 0).values[0]
    return float(best_record[0]), float(best_record[1]), tuple(
        best_record[2:ldope.SPHERICAL_INDIVIDUAL_SIZE + 2])


def plot_spherical_result_by_individual(case_index,
                                        individual,
                                        fig,
                                        norm=True,
                                        p_color=seaborn.xkcd_rgb['red'],
                                        e_color=seaborn.xkcd_rgb['green']):
    state_p0 = solve_spherical_model.CASES[case_index - 1][0]
    state_e0 = solve_spherical_model.CASES[case_index - 1][1]
    if norm:
        state_p0 = ldope.spherical_state_norm_fcn(state_p0)
        state_e0 = ldope.spherical_state_norm_fcn(state_e0)

    costate_p0, costate_e0, tf_norm = ldope.spherical_individual_convert_fcn(
        individual)

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

    result = solve_ivp(
        lambda t, y: ldope.spherical_ext_state_fcn(tuple(y.tolist()), norm),
        t_span=np.array(t_span),
        y0=np.array(state_p0 + state_e0 + costate_p0 + costate_e0),
        t_eval=t_eval)

    xp, yp, zp = [], [], []
    xe, ye, ze = [], [], []
    r = []

    for i in range(len(result.t)):
        split_state = result.y.T[i]
        r_p, r_e = float(split_state[0]), float(split_state[6])
        xi_p, xi_e = float(split_state[3]), float(split_state[9])
        phi_p, phi_e = float(split_state[4]), float(split_state[10])
        xp.append(r_p * cos(phi_p) * cos(xi_p))
        yp.append(r_p * cos(phi_p) * sin(xi_p))
        zp.append(r_p * sin(phi_p))
        xe.append(r_e * cos(phi_e) * cos(xi_e))
        ye.append(r_e * cos(phi_e) * sin(xi_e))
        ze.append(r_e * sin(phi_e))
        r.append(
            sqrt((xp[-1] - xe[-1])**2 + (yp[-1] - ye[-1])**2 +
                 (zp[-1] - ze[-1])**2))

    ax = fig.gca(projection='3d')

    ax.scatter(xp[0],
               yp[0],
               zp[0],
               color=p_color,
               marker='*',
               s=50,
               label='Initial Position P')
    ax.scatter(xe[0],
               ye[0],
               ze[0],
               color=e_color,
               marker='s',
               s=25,
               label='Initial Position E')
    ax.plot(xp, yp, zp, '-', color=p_color, label='Pursuer Trajectory')
    ax.plot(xe, ye, ze, '--', color=e_color, label='Evader Trajectory')
    ax.legend()

    return fig


if __name__ == "__main__":

    case_index = 1
    _, _, individual_norm = find_best_spherical_record(case_index, True)
    _, _, individual = find_best_spherical_record(case_index, False)

    fig_norm = plot_util.plot_norm_frame(
        'case {} norm frame'.format(case_index))
    fig = plot_util.plot_frame('case {} frame'.format(case_index))

    plot_spherical_result_by_individual(case_index, individual_norm, fig_norm,
                                        True)
    plot_spherical_result_by_individual(case_index, individual, fig, False)

    plt.show()
