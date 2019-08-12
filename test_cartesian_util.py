#%% import
import sys
import ctypes
import numpy as np
from numpy import deg2rad, sqrt

#%% 加载dll
platform = sys.platform
print(platform)

if platform == 'darwin':
    ldope = ctypes.CDLL('./lib/libldope.dylib')

print(ldope)

#%% 测试导出常量
get_cartesian_state_size = ldope.get_cartesian_state_size
cartesian_state_size_c = ctypes.c_size_t()
get_cartesian_state_size(ctypes.byref(cartesian_state_size_c))
cartesian_state_size = cartesian_state_size_c.value

print('cartesian state size:')
print(cartesian_state_size)

get_cartesian_control_size = ldope.get_cartesian_control_size
cartesian_control_size_c = ctypes.c_size_t()
get_cartesian_control_size(ctypes.byref(cartesian_control_size_c))
cartesian_control_size = cartesian_control_size_c.value

print('cartesian control size:')
print(cartesian_control_size)

get_cartesian_ext_state_size = ldope.get_cartesian_ext_state_size
cartesian_ext_state_size_c = ctypes.c_size_t()
get_cartesian_ext_state_size(ctypes.byref(cartesian_ext_state_size_c))
cartesian_ext_state_size = cartesian_ext_state_size_c.value

print('cartesian ext state size:')
print(cartesian_ext_state_size)

get_cartesian_boundary_size = ldope.get_cartesian_boundary_size
cartesian_boundary_size_c = ctypes.c_size_t()
get_cartesian_boundary_size(ctypes.byref(cartesian_boundary_size_c))
cartesian_boundary_size = cartesian_boundary_size_c.value

print('cartesian boundary size:')
print(cartesian_boundary_size)

get_cartesian_individual_size = ldope.get_cartesian_individual_size
cartesian_individual_size_c = ctypes.c_size_t()
get_cartesian_individual_size(ctypes.byref(cartesian_individual_size_c))
cartesian_individual_size = cartesian_individual_size_c.value

print('cartesian individual size:')
print(cartesian_individual_size)

#%% 测试笛卡尔坐标系个体转换
util_cartesian_converter = ldope.cartesian_individual_convert_fcn
print(util_cartesian_converter)

individual_type = ctypes.c_double * cartesian_individual_size
individual_c = individual_type(1.3953, -7.1342, -3.4850, 6.1483, 5.1990,
                               2.3712, 4.2746, 0.2502, 1.0205, -3.2339,
                               -3.1354, -7.0502, 3.3174)

costate_type = ctypes.c_double * cartesian_state_size
costate_p_c = costate_type()
costate_e_c = costate_type()

tf_c = ctypes.c_double()

util_cartesian_converter(individual_c, costate_p_c, costate_e_c,
                         ctypes.byref(tf_c))

costate_p = tuple(costate_p_c)
costate_e = tuple(costate_e_c)
tf = tf_c.value

print('costate p:')
print(costate_p)

print('costate e:')
print(costate_e)

print('tf:')
print(tf)

#%% 测试笛卡尔坐标系归一化和反归一化
util_cartesian_norm = ldope.cartesian_state_norm_fcn
print(util_cartesian_norm)

state_type = ctypes.c_double * cartesian_state_size
state_c = state_type(-6598.210e3, 1163.440e3, 0, 1.160e3, 6.578e3, 3.857e3)

du = 6378.137e3
mu = 3.986004418e14
tu = sqrt(du**3 / mu)

normed_state_c = state_type()

util_cartesian_norm(state_c, ctypes.c_double(du), ctypes.c_double(tu),
                    normed_state_c)

normed_state = tuple(normed_state_c)

print('normed state:')
print(normed_state)

util_cartesian_denorm = ldope.cartesian_state_denorm_fcn
print(util_cartesian_denorm)

denormed_state_c = state_type()

util_cartesian_denorm(normed_state_c, ctypes.c_double(du), ctypes.c_double(tu),
                      denormed_state_c)

denormed_state = tuple(denormed_state_c)

print('denormed state:')
print(denormed_state)

#%%
