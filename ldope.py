#%% import
import sys
import ctypes
import numpy as np
from numpy import deg2rad, sqrt
import time

#%% 加载dll
platform = sys.platform
print(platform)

if (platform == 'darwin'):
    ldope = ctypes.CDLL('./lib/libldope.dylib')

print(ldope)

#%% 常量
DU = 6378.137e3
MU = 3.986004418e14
TU = sqrt(DU**3 / MU)
VU = DU / TU
GU = VU / TU

TM_P = 0.1 * GU
TM_E = 0.05 * GU

SPHERICAL_K = (1, DU, DU, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
CARTESIAN_K = (10, 10, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)

get_spherical_state_size = ldope.get_spherical_state_size
spherical_state_size_c = ctypes.c_size_t()
get_spherical_state_size(ctypes.byref(spherical_state_size_c))
SPHERICAL_STATE_SIZE = spherical_state_size_c.value

# print('spherical state size:')
# print(SPHERICAL_STATE_SIZE)

get_spherical_control_size = ldope.get_spherical_control_size
spherical_control_size_c = ctypes.c_size_t()
get_spherical_control_size(ctypes.byref(spherical_control_size_c))
SPHERICAL_CONTROL_SIZE = spherical_control_size_c.value

# print('spherical control size:')
# print(SPHERICAL_CONTROL_SIZE)

get_spherical_ext_state_size = ldope.get_spherical_ext_state_size
spherical_ext_state_size_c = ctypes.c_size_t()
get_spherical_ext_state_size(ctypes.byref(spherical_ext_state_size_c))
SPHERICAL_EXT_STATE_SIZE = spherical_ext_state_size_c.value

# print('spherical ext state size:')
# print(SPHERICAL_EXT_STATE_SIZE)

get_spherical_boundary_size = ldope.get_spherical_boundary_size
spherical_boundary_size_c = ctypes.c_size_t()
get_spherical_boundary_size(ctypes.byref(spherical_boundary_size_c))
SPHERICAL_BOUNDARY_SIZE = spherical_boundary_size_c.value

# print('spherical boundary size:')
# print(SPHERICAL_BOUNDARY_SIZE)

get_spherical_individual_size = ldope.get_spherical_individual_size
spherical_individual_size_c = ctypes.c_size_t()
get_spherical_individual_size(ctypes.byref(spherical_individual_size_c))
SPHERICAL_INDIVIDUAL_SIZE = spherical_individual_size_c.value

# print('spherical individual size:')
# print(SPHERICAL_INDIVIDUAL_SIZE)

get_cartesian_state_size = ldope.get_cartesian_state_size
cartesian_state_size_c = ctypes.c_size_t()
get_cartesian_state_size(ctypes.byref(cartesian_state_size_c))
CARTESIAN_STATE_SIZE = cartesian_state_size_c.value

# print('cartesian state size:')
# print(CARTESIAN_STATE_SIZE)

get_cartesian_control_size = ldope.get_cartesian_control_size
cartesian_control_size_c = ctypes.c_size_t()
get_cartesian_control_size(ctypes.byref(cartesian_control_size_c))
CARTESIAN_CONTROL_SIZE = cartesian_control_size_c.value

# print('cartesian control size:')
# print(CARTESIAN_CONTROL_SIZE)

get_cartesian_ext_state_size = ldope.get_cartesian_ext_state_size
cartesian_ext_state_size_c = ctypes.c_size_t()
get_cartesian_ext_state_size(ctypes.byref(cartesian_ext_state_size_c))
CARTESIAN_EXT_STATE_SIZE = cartesian_ext_state_size_c.value

# print('cartesian ext state size:')
# print(CARTESIAN_EXT_STATE_SIZE)

get_cartesian_boundary_size = ldope.get_cartesian_boundary_size
cartesian_boundary_size_c = ctypes.c_size_t()
get_cartesian_boundary_size(ctypes.byref(cartesian_boundary_size_c))
CARTESIAN_BOUNDARY_SIZE = cartesian_boundary_size_c.value

# print('cartesian boundary size:')
# print(CARTESIAN_BOUNDARY_SIZE)

get_cartesian_individual_size = ldope.get_cartesian_individual_size
cartesian_individual_size_c = ctypes.c_size_t()
get_cartesian_individual_size(ctypes.byref(cartesian_individual_size_c))
CARTESIAN_INDIVIDUAL_SIZE = cartesian_individual_size_c.value

# print('cartesian individual size:')
# print(CARTESIAN_INDIVIDUAL_SIZE)

spherical_state_type = ctypes.c_double * SPHERICAL_STATE_SIZE
spherical_costate_type = ctypes.c_double * SPHERICAL_STATE_SIZE
spherical_control_type = ctypes.c_double * SPHERICAL_CONTROL_SIZE
spherical_ext_state_type = ctypes.c_double * SPHERICAL_EXT_STATE_SIZE
spherical_boundary_type = ctypes.c_double * SPHERICAL_BOUNDARY_SIZE
spherical_individual_type = ctypes.c_double * SPHERICAL_INDIVIDUAL_SIZE

cartesian_state_type = ctypes.c_double * CARTESIAN_STATE_SIZE
cartesian_costate_type = ctypes.c_double * CARTESIAN_STATE_SIZE
cartesian_control_type = ctypes.c_double * CARTESIAN_CONTROL_SIZE
cartesian_ext_state_type = ctypes.c_double * CARTESIAN_EXT_STATE_SIZE
cartesian_boundary_type = ctypes.c_double * CARTESIAN_BOUNDARY_SIZE
cartesian_individual_type = ctypes.c_double * CARTESIAN_INDIVIDUAL_SIZE

du_c = ctypes.c_double(DU)
mu_c = ctypes.c_double(MU)
tu_c = ctypes.c_double(TU)

tm_p_c = ctypes.c_double(TM_P)
tm_e_c = ctypes.c_double(TM_E)
tm_p_c_norm = ctypes.c_double(TM_P / GU)
tm_e_c_norm = ctypes.c_double(TM_E / GU)


#%% 辅助函数的定义
def spherical_individual_convert_fcn(individual):
    individual_c = spherical_individual_type(*individual)
    costate_p_c = spherical_costate_type()
    costate_e_c = spherical_costate_type()
    tf_c = ctypes.c_double()
    ldope.spherical_individual_convert_fcn(individual_c, costate_p_c,
                                           costate_e_c, ctypes.byref(tf_c))
    return tuple(costate_p_c), tuple(costate_e_c), tf_c.value


def spherical_state_norm_fcn(state):
    state_c = spherical_state_type(*state)
    normed_state_c = spherical_state_type()
    ldope.spherical_state_norm_fcn(state_c, du_c, tu_c, normed_state_c)
    return tuple(normed_state_c)


def cartesian_individual_convert_fcn(individual):
    individual_c = cartesian_individual_type(*individual)
    costate_p_c = cartesian_costate_type()
    costate_e_c = cartesian_costate_type()
    tf_c = ctypes.c_double()
    ldope.cartesian_individual_convert_fcn(individual_c, costate_p_c,
                                           costate_e_c, ctypes.byref(tf_c))
    return tuple(costate_p_c), tuple(costate_e_c), tf_c.value


def cartesian_state_norm_fcn(state):
    state_c = cartesian_state_type(*state)
    normed_state_c = cartesian_state_type()
    ldope.cartesian_state_norm_fcn(state_c, du_c, tu_c, normed_state_c)
    return tuple(normed_state_c)


#%% 核心函数的定义
def spherical_ext_state_fcn(ext_state, normalized=False):
    ext_state_c = spherical_ext_state_type(*ext_state)
    dot_ext_state_c = spherical_ext_state_type()

    if normalized:
        ldope.spherical_ext_state_fcn(ext_state_c, tm_p_c_norm, tm_e_c_norm,
                                      ctypes.c_double(1), dot_ext_state_c)
    else:
        ldope.spherical_ext_state_fcn(ext_state_c, tm_p_c, tm_e_c, mu_c,
                                      dot_ext_state_c)
    return tuple(dot_ext_state_c)


def cartesian_ext_state_fcn(ext_state, normalized=False):
    ext_state_c = cartesian_ext_state_type(*ext_state)
    dot_ext_state_c = cartesian_ext_state_type()

    if normalized:
        ldope.cartesian_ext_state_fcn(ext_state_c, tm_p_c_norm, tm_e_c_norm,
                                      ctypes.c_double(1), dot_ext_state_c)
    else:
        ldope.cartesian_ext_state_fcn(ext_state_c, tm_p_c, tm_e_c, mu_c,
                                      dot_ext_state_c)
    return tuple(dot_ext_state_c)


#%% 求解函数的定义
class Param(ctypes.Structure):
    _fields_ = [('pInitialStateP', ctypes.POINTER(ctypes.c_double)),
                ('pInitialStateE', ctypes.POINTER(ctypes.c_double)),
                ('tmP', ctypes.c_double), ('tmE', ctypes.c_double),
                ('lb', ctypes.POINTER(ctypes.c_double)),
                ('ub', ctypes.POINTER(ctypes.c_double)),
                ('pK', ctypes.POINTER(ctypes.c_double)),
                ('du', ctypes.c_double), ('tu', ctypes.c_double),
                ('printProcess', ctypes.c_bool)]


def spherical_solve_fcn(initial_state_p,
                        initial_state_e,
                        lb,
                        ub,
                        print_process=True):
    initial_state_p_c = spherical_state_type(*initial_state_p)
    initial_state_e_c = spherical_state_type(*initial_state_e)
    lb_c = spherical_individual_type(*lb)
    ub_c = spherical_individual_type(*ub)
    k_c = spherical_boundary_type(*SPHERICAL_K)
    param = Param(pInitialStateP=initial_state_p_c,
                  pInitialStateE=initial_state_e_c,
                  tmP=TM_P,
                  tmE=TM_E,
                  lb=lb_c,
                  ub=ub_c,
                  pK=k_c,
                  du=DU,
                  tu=TU,
                  printProcess=print_process)
    opt_individual_c = spherical_individual_type()
    opt_value_c = ctypes.c_double()

    start_time = time.perf_counter()
    ldope.spherical_solve_fcn(ctypes.byref(param), opt_individual_c,
                              ctypes.byref(opt_value_c))
    end_time = time.perf_counter()

    return (tuple(opt_individual_c), opt_value_c.value, end_time - start_time)


def cartesian_solve_fcn(initial_state_p,
                        initial_state_e,
                        lb,
                        ub,
                        print_process=True):
    initial_state_p_c = cartesian_state_type(*initial_state_p)
    initial_state_e_c = cartesian_state_type(*initial_state_e)
    lb_c = cartesian_individual_type(*lb)
    ub_c = cartesian_individual_type(*ub)
    k_c = cartesian_boundary_type(*CARTESIAN_K)
    param = Param(pInitialStateP=initial_state_p_c,
                  pInitialStateE=initial_state_e_c,
                  tmP=TM_P,
                  tmE=TM_E,
                  lb=lb_c,
                  ub=ub_c,
                  pK=k_c,
                  du=DU,
                  tu=TU,
                  printProcess=print_process)
    opt_individual_c = cartesian_individual_type()
    opt_value_c = ctypes.c_double()

    start_time = time.perf_counter()
    ldope.cartesian_solve_fcn(ctypes.byref(param), opt_individual_c,
                              ctypes.byref(opt_value_c))
    end_time = time.perf_counter()

    return (tuple(opt_individual_c), opt_value_c.value, end_time - start_time)
