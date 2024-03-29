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
elif platform == 'win32':
    ldope = ctypes.CDLL('./lib/libldope.dll')

print(ldope)

#%% 初始状态

du = 6378.137e3
mu = 3.986004418e14
tu = sqrt(du**3 / mu)
vu = du / tu
gu = vu / tu

tm_p = 0.1 * gu
tm_e = 0.05 * gu

state_p0 = (-6598.210e3, 1163.440e3, 0, -0.458e3, -2.598e3, 7.248e3)
state_e0 = (-6755e3, -3900e3, 0, 3.520e3, -6.097e3, 1.241e3)
costate_p0 = (1.3953, -7.1342, -3.4850, 6.1483, 5.1990, 2.3712)
costate_e0 = (4.2746, 0.2502, 1.0205, -3.2339, -3.1354, -7.0502)
tf = 3.3174

individual = (1.3953, -7.1342, -3.4850, 6.1483, 5.1990, 2.3712, 4.2746, 0.2502,
              1.0205, -3.2339, -3.1354, -7.0502, 3.3174)
k = (10, 10, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)

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

state_type = ctypes.c_double * cartesian_state_size
state_p0_c = state_type(*state_p0)
state_e0_c = state_type(*state_e0)

costate_type = ctypes.c_double * cartesian_state_size
costate_p0_c = costate_type(*costate_p0)
costate_e0_c = costate_type(*costate_e0)

control_type = ctypes.c_double * cartesian_control_size
control_p0_c = control_type()
control_e0_c = control_type()

ext_state_type = ctypes.c_double * cartesian_ext_state_size
ext_state_c = ext_state_type(*state_p0, *state_e0, *costate_p0, *costate_e0)

boundary_type = ctypes.c_double * cartesian_boundary_size

du_c = ctypes.c_double(du)
tu_c = ctypes.c_double(tu)

tm_p_c = ctypes.c_double(tm_p)
tm_e_c = ctypes.c_double(tm_e)

mu_c = ctypes.c_double(mu)

individual_type = ctypes.c_double * cartesian_individual_size
individual_c = individual_type(*individual)

k_type = ctypes.c_double * cartesian_boundary_size
k_c = k_type(*k)

#%% 球坐标系状态微分方程
cartesian_state_fcn = ldope.cartesian_state_fcn

dot_state_p_c = state_type()
dot_state_e_c = state_type()

cartesian_state_fcn(state_p0_c, control_p0_c, tm_p_c, mu_c, dot_state_p_c)
cartesian_state_fcn(state_e0_c, control_e0_c, tm_e_c, mu_c, dot_state_e_c)

dot_state_p = tuple(dot_state_p_c)
dot_state_e = tuple(dot_state_e_c)

print('dot state p:')
print(dot_state_p)

print('dot state e:')
print(dot_state_e)

#%% 球坐标系协态微分方程
cartesian_costate_fcn = ldope.cartesian_costate_fcn

dot_costate_p_c = costate_type()
dot_costate_e_c = costate_type()

cartesian_costate_fcn(state_p0_c, costate_p0_c, control_p0_c, tm_p_c, mu_c,
                      dot_costate_p_c)
cartesian_costate_fcn(state_e0_c, costate_e0_c, control_e0_c, tm_e_c, mu_c,
                      dot_costate_e_c)

dot_costate_p = tuple(dot_costate_p_c)
dot_costate_e = tuple(dot_costate_e_c)

print('dot costate p:')
print(dot_costate_p)

print('dot costate e:')
print(dot_costate_e)

#%% 球坐标系控制变量函数
cartesian_control_fcn = ldope.cartesian_control_fcn

flag_p_c = ctypes.c_int(-1)
flag_e_c = ctypes.c_int(1)

control_p_c = control_type()
control_e_c = control_type()

cartesian_control_fcn(state_p0_c, costate_p0_c, flag_p_c, control_p_c)
cartesian_control_fcn(state_e0_c, costate_e0_c, flag_e_c, control_e_c)

control_p = tuple(control_p_c)
control_e = tuple(control_e_c)

print('control p:')
print(control_p)

print('control e:')
print(control_e)

#%% 球坐标系哈密顿函数
cartesian_hamilton_fcn = ldope.cartesian_hamilton_fcn

hamilton_p_c = ctypes.c_double()
hamilton_e_c = ctypes.c_double()

cartesian_hamilton_fcn(state_p0_c, costate_p0_c, control_p0_c, tm_p_c, mu_c,
                       ctypes.byref(hamilton_p_c))
cartesian_hamilton_fcn(state_e0_c, costate_e0_c, control_e0_c, tm_e_c, mu_c,
                       ctypes.byref(hamilton_e_c))

hamilton_p = hamilton_p_c.value
hamilton_e = hamilton_e_c.value

print('hamilton p:')
print(hamilton_p)

print('hamilton e:')
print(hamilton_e)

#%% 球坐标系扩展状态微分方程
cartesian_ext_state_fcn = ldope.cartesian_ext_state_fcn

dot_ext_state_c = ext_state_type()

cartesian_ext_state_fcn(ext_state_c, tm_p_c, tm_e_c, mu_c, dot_ext_state_c)

dot_ext_state = tuple(dot_ext_state_c)

print('dot ext state:')
print(dot_ext_state)

#%% 球坐标系边界条件
cartesian_boundary_fcn = ldope.cartesian_boundary_fcn

boundary_c = boundary_type()

cartesian_boundary_fcn(ext_state_c, tm_p_c, tm_e_c, mu_c, boundary_c)

boundary = tuple(boundary_c)

print('boundary:')
print(boundary)

#%% 球坐标系适应度函数
cartesian_fitness_fcn = ldope.cartesian_fitness_fcn

fitness_c = ctypes.c_double()

cartesian_fitness_fcn(individual_c, k_c, state_p0_c, state_e0_c, tm_p_c,
                      tm_e_c, du_c, tu_c, ctypes.byref(fitness_c))

fitness = fitness_c.value

print('fitness:')
print(fitness)
