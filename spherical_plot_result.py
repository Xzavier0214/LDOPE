#%% import
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy import sin, cos, sqrt
import seaborn
from scipy.integrate import solve_ivp
import ldope
import spherical_solve

#%% 初始化
case_index = 3
index = 1

state_p0 = spherical_solve.CASES[case_index - 1][0]
state_e0 = spherical_solve.CASES[case_index - 1][1]
state_p0_norm = ldope.spherical_state_norm_fcn(state_p0)
state_e0_norm = ldope.spherical_state_norm_fcn(state_e0)

read_line = 1
with open('./data/spherical/{}.data'.format(case_index), 'r') as data:
    for each_line in data:
        if read_line < index:
            read_line += 1
            continue
        temp = each_line.split(' ')
        individual = [
            float(ele) for ele in temp[2:ldope.SPHERICAL_INDIVIDUAL_SIZE + 2]
        ]
        opt_value = float(temp[0])
        time_cost = float(temp[1])
        break

costate_p0, costate_e0, tf_norm = ldope.spherical_individual_convert_fcn(
    individual)

#%% 计算归一化结果
step = 0.05
t_span = 0, tf_norm
t_eval = np.arange(0, tf_norm, step)
t_eval = np.append(t_eval, tf_norm)

result = solve_ivp(
    lambda t, y: ldope.spherical_ext_state_fcn(tuple(y.tolist()), True),
    t_span=np.array(t_span),
    y0=np.array(state_p0_norm + state_e0_norm + costate_p0 + costate_e0),
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

#%% 绘制归一化结果
fig_norm = plt.figure(1)
ax = fig_norm.gca(projection='3d')

ax.scatter(0,
           0,
           0,
           color=seaborn.xkcd_rgb['black'],
           s=50,
           label='Earth Center')

ax.set_xlabel(r'$\bar{x}$' + ' (normalized x)')
ax.set_ylabel(r'$\bar{y}$' + ' (normalized y)')
ax.set_zlabel(r'$\bar{z}$' + ' (normalized z)')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)

ax.scatter(xp[0],
           yp[0],
           zp[0],
           color='r',
           marker='*',
           s=50,
           label='Initial Position P')
ax.scatter(xe[0],
           ye[0],
           ze[0],
           color='g',
           marker='s',
           s=25,
           label='Initial Position E')
ax.plot(xp, yp, zp, 'r-', label='Pursuer Trajectory')
ax.plot(xe, ye, ze, 'g--', label='Evader Trajectory')
ax.legend()

#%% 计算常规结果
step = 0.05 * ldope.TU
t_span = 0, tf_norm * ldope.TU
t_eval = np.arange(0, tf_norm * ldope.TU, step)
t_eval = np.append(t_eval, tf_norm * ldope.TU)

result = solve_ivp(
    lambda t, y: ldope.spherical_ext_state_fcn(tuple(y.tolist()), False),
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

#%% 绘制常规结果
fig = plt.figure(2)
ax = fig.gca(projection='3d')

ax.scatter(0,
           0,
           0,
           color=seaborn.xkcd_rgb['black'],
           s=50,
           label='Earth Center')

ax.set_xlabel(r'$\bar{x}$' + ' (x)')
ax.set_ylabel(r'$\bar{y}$' + ' (y)')
ax.set_zlabel(r'$\bar{z}$' + ' (z)')
ax.set_xlim(-1 * ldope.DU, 1 * ldope.DU)
ax.set_ylim(-1 * ldope.DU, 1 * ldope.DU)
ax.set_zlim(-1 * ldope.DU, 1 * ldope.DU)

ax.scatter(xp[0],
           yp[0],
           zp[0],
           color='r',
           marker='*',
           s=50,
           label='Initial Position P')
ax.scatter(xe[0],
           ye[0],
           ze[0],
           color='g',
           marker='s',
           s=25,
           label='Initial Position E')
ax.plot(xp, yp, zp, 'r-', label='Pursuer Trajectory')
ax.plot(xe, ye, ze, 'g--', label='Evader Trajectory')
ax.legend()

plt.show()
