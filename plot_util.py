import matplotlib.pyplot as plt
from matplotlib import ticker
from mpl_toolkits.mplot3d import Axes3D
import seaborn
import ldope


# 绘制3D坐标轴
def plot_3D_frame(num, norm=True):
    fig = plt.figure(num)
    ax = fig.gca(projection='3d')

    # 绘制地球
    ax.scatter(0,
               0,
               0,
               color=seaborn.xkcd_rgb['black'],
               s=50,
               label='Earth Center')

    if norm:
        ax.set_xlabel(r'$\bar{x}$' + ' (normalized x)')
        ax.set_ylabel(r'$\bar{y}$' + ' (normalized y)')
        ax.set_zlabel(r'$\bar{z}$' + ' (normalized z)')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
    else:
        ax.set_xlabel(r'$x$' + r' ($x10^6$)')
        ax.set_ylabel(r'$y$' + r' ($x10^6$)')
        ax.set_zlabel(r'$z$' + r' ($x10^6$)')
        ax.set_xlim(-1 * ldope.DU, 1 * ldope.DU)
        ax.set_ylim(-1 * ldope.DU, 1 * ldope.DU)
        ax.set_zlim(-1 * ldope.DU, 1 * ldope.DU)

    if not norm:

        def formatnum(x, pos):
            return '$%.2f$' % (x / 1e6)

        formatter = ticker.FuncFormatter(formatnum)

        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        ax.zaxis.set_major_formatter(formatter)

    return fig, ax


# 绘制控制变量坐标轴
def plot_control_frame(num):
    fig_control, (ax_control_alpha, ax_control_beta) = plt.subplots(2,
                                                                    1,
                                                                    num=num)

    ax2_alpha = ax_control_alpha.twinx()
    ax2_beta = ax_control_beta.twinx()

    ax_control_alpha.set_xlabel(r'$\bar{t}$' + r' (normalized time)')
    ax_control_beta.set_xlabel(r'$\bar{t}$' + r' (normalized time)')

    ax_control_alpha.grid()
    ax_control_beta.grid()

    return fig_control, (ax_control_alpha, ax2_alpha, ax_control_beta,
                         ax2_beta)


# 绘制时间-距离坐标轴
def plot_td_frame(num, norm=True):
    fig = plt.figure(num)
    ax = fig.gca()

    if norm:
        ax.set_xlabel(r'$\bar{t}$' + r' (normalized time)')
        ax.set_ylabel(r'$\bar{D}$' + r' (normalized distance)')
    else:
        ax.set_xlabel(r'$t$' + r' (time)')
        ax.set_ylabel(r'$D$' + r' (distance)')

    if not norm:
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 2))

        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)

    ax.grid()

    return fig, ax


# 绘制时间-半径坐标轴
def plot_tr_frame(num):
    fig, (ax_p, ax_e) = plt.subplots(2, 1, num=num)

    ax_p.set_xlabel(r'$\bar{t}$' + r' (normalized time)')
    ax_p.set_ylabel(r'$\bar{r}$' + r' (normalized radius)')
    ax_e.set_xlabel(r'$\bar{t}$' + r' (normalized time)')
    ax_e.set_ylabel(r'$\bar{r}$' + r' (normalized radius)')

    return fig, (ax_p, ax_e)


# 绘制时间-纬度坐标轴
def plot_tphi_frame(num):
    fig, (ax_p, ax_e) = plt.subplots(2, 1, num=num)

    ax_p.set_xlabel(r'$\bar{t}$' + r' (normalized time)')
    ax_p.set_ylabel(r'$\varphi$' + r' (latitude)')
    ax_e.set_xlabel(r'$\bar{t}$' + r' (normalized time)')
    ax_e.set_ylabel(r'$\varphi$' + r' (latitude)')

    return fig, (ax_p, ax_e)


# 绘制时间-经度坐标轴
def plot_txi_frame(num, modified=False):
    fig, (ax_p, ax_e) = plt.subplots(2, 1, num=num)

    if modified:
        ax_p.set_xlabel(r'$\bar{t}$' + r' (normalized time)')
        ax_p.set_ylabel(r'$\hat{\xi}$' + r' (absolute longitude)')
        ax_e.set_xlabel(r'$\bar{t}$' + r' (normalized time)')
        ax_e.set_ylabel(r'$\hat{\xi}$' + r' (absolute longitude)')
    else:
        ax_p.set_xlabel(r'$\bar{t}$' + r' (normalized time)')
        ax_p.set_ylabel(r'$\xi$' + r' (absolute longitude)')
        ax_e.set_xlabel(r'$\bar{t}$' + r' (normalized time)')
        ax_e.set_ylabel(r'$\xi$' + r' (absolute longitude)')

    return fig, (ax_p, ax_e)
