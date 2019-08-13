import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn
import ldope


def plot_norm_frame(fig_name):
    fig_norm = plt.figure(fig_name)
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

    return fig_norm


def plot_frame(fig_name):
    fig = plt.figure(fig_name)
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

    return fig
