#%% import
import sys
import ctypes
import numpy as np
from numpy import deg2rad, sqrt

#%% 加载dll
platform = sys.platform
print(platform)

if (platform == 'darwin'):
    ldope = ctypes.CDLL('./lib/libldope.dylib')

print(ldope)

#%% 测试导出常量
get_spherical_state_size = ldope.get_spherical_state_size
get_spherical_state_size.restype = ctypes.c_size_t
spherical_state_size = get_spherical_state_size()

print('spherical state size:')
print(spherical_state_size)

get_spherical_control_size = ldope.get_spherical_control_size
get_spherical_control_size.restype = ctypes.c_size_t
spherical_control_size = get_spherical_control_size()

print('spherical control size:')
print(spherical_control_size)

get_spherical_ext_state_size = ldope.get_spherical_ext_state_size
get_spherical_ext_state_size.restype = ctypes.c_size_t
spherical_ext_state_size = get_spherical_ext_state_size()

print('spherical ext state size:')
print(spherical_ext_state_size)

get_spherical_boundary_size = ldope.get_spherical_boundary_size
get_spherical_boundary_size.restype = ctypes.c_size_t
spherical_boundary_size = get_spherical_boundary_size()

print('spherical boundary size:')
print(spherical_boundary_size)

get_spherical_individual_size = ldope.get_spherical_individual_size
get_spherical_individual_size.restype = ctypes.c_size_t
spherical_individual_size = get_spherical_individual_size()

print('spherical individual size:')
print(spherical_individual_size)

#%% 测试球坐标系个体转换
util_spherical_converter = ldope.spherical_individual_convert_fcn
print(util_spherical_converter)

individual_type = ctypes.c_double * spherical_individual_size
individual_c = individual_type(-0.2072, -9.2840, -9.9743, -2.1281, 1.7394,
                               -9.0355, 9.0326, 9.2870, 3.6799, 6.5160, 1.4810)

costate_type = ctypes.c_double * spherical_state_size
costate_p_c = costate_type()
costate_e_c = costate_type()

tf_c = ctypes.c_double()

util_spherical_converter(individual_c, costate_p_c, costate_e_c,
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

#%%
util_spherical_norm = ldope.spherical_state_norm_fcn
print(util_spherical_norm)

state_type = ctypes.c_double * spherical_state_size
state_c = state_type(6700e3, 7.713e3, deg2rad(0), deg2rad(170), deg2rad(0),
                     deg2rad(70))

du = 6378.137e3
mu = 3.986004418e14
tu = sqrt(du**3 / mu)

normed_state_c = state_type()

util_spherical_norm(state_c, ctypes.c_double(du), ctypes.c_double(tu),
                    normed_state_c)

normed_state = tuple(normed_state_c)

print('normed state:')
print(normed_state)

util_spherical_denorm = ldope.spherical_state_denorm_fcn
print(util_spherical_denorm)

denormed_state_c = state_type()

util_spherical_denorm(normed_state_c, ctypes.c_double(du), ctypes.c_double(tu),
                      denormed_state_c)

denormed_state = tuple(denormed_state_c)

print('denormed state:')
print(denormed_state)

#%%
