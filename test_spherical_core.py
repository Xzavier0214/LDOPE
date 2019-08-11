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

#%% 初始状态

du = 6378.137e3
mu = 3.986004418e14
tu = sqrt(du**3 / mu)
vu = du / tu
gu = vu / tu

tm_p = 0.1 * gu
tm_e = 0.05 * gu

state_p0 = (6700e3, 7.713e3, deg2rad(0), deg2rad(170), deg2rad(0), deg2rad(70))
state_e0 = (7800e3, 7.149e3, deg2rad(0), deg2rad(210), deg2rad(0), deg2rad(10))
costate_p0 = (-9.0921, 0.8132, 5.6422, 0, 1.6151, -7.0374)
costate_e0 = (6.0629, 5.5236, -8.5717, 0, -2.2229, -2.3348)
tf = 3.220

individual = (-9.0921, 0.8132, 5.6422, 1.6151, -7.0374, 6.0629, 5.5236,
              -8.5717, -2.2229, -2.3348, 3.220)
k = (1, du, du, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)

get_spherical_state_size = ldope.get_spherical_state_size
spherical_state_size_c = ctypes.c_size_t()
get_spherical_state_size(ctypes.byref(spherical_state_size_c))
spherical_state_size = spherical_state_size_c.value

print('spherical state size:')
print(spherical_state_size)

get_spherical_control_size = ldope.get_spherical_control_size
spherical_control_size_c = ctypes.c_size_t()
get_spherical_control_size(ctypes.byref(spherical_control_size_c))
spherical_control_size = spherical_control_size_c.value

print('spherical control size:')
print(spherical_control_size)

get_spherical_ext_state_size = ldope.get_spherical_ext_state_size
spherical_ext_state_size_c = ctypes.c_size_t()
get_spherical_ext_state_size(ctypes.byref(spherical_ext_state_size_c))
spherical_ext_state_size = spherical_ext_state_size_c.value

print('spherical ext state size:')
print(spherical_ext_state_size)

get_spherical_boundary_size = ldope.get_spherical_boundary_size
spherical_boundary_size_c = ctypes.c_size_t()
get_spherical_boundary_size(ctypes.byref(spherical_boundary_size_c))
spherical_boundary_size = spherical_boundary_size_c.value

print('spherical boundary size:')
print(spherical_boundary_size)

get_spherical_individual_size = ldope.get_spherical_individual_size
spherical_individual_size_c = ctypes.c_size_t()
get_spherical_individual_size(ctypes.byref(spherical_individual_size_c))
spherical_individual_size = spherical_individual_size_c.value

print('spherical individual size:')
print(spherical_individual_size)

state_type = ctypes.c_double * spherical_state_size
state_p0_c = state_type(*state_p0)
state_e0_c = state_type(*state_e0)

costate_type = ctypes.c_double * spherical_state_size
costate_p0_c = costate_type(*costate_p0)
costate_e0_c = costate_type(*costate_e0)

control_type = ctypes.c_double * spherical_control_size
control_p0_c = control_type()
control_e0_c = control_type()

ext_state_type = ctypes.c_double * spherical_ext_state_size
ext_state_c = ext_state_type(*state_p0, *state_e0, *costate_p0, *costate_e0)

boundary_type = ctypes.c_double * spherical_boundary_size

du_c = ctypes.c_double(du)
tu_c = ctypes.c_double(tu)

tm_p_c = ctypes.c_double(tm_p)
tm_e_c = ctypes.c_double(tm_e)

mu_c = ctypes.c_double(mu)

individual_type = ctypes.c_double * spherical_individual_size
individual_c = individual_type(*individual)

k_type = ctypes.c_double * spherical_boundary_size
k_c = k_type(*k)

#%% 球坐标系状态微分方程
spherical_state_fcn = ldope.spherical_state_fcn

dot_state_p_c = state_type()
dot_state_e_c = state_type()

spherical_state_fcn(state_p0_c, control_p0_c, tm_p_c, mu_c, dot_state_p_c)
spherical_state_fcn(state_e0_c, control_e0_c, tm_e_c, mu_c, dot_state_e_c)

dot_state_p = tuple(dot_state_p_c)
dot_state_e = tuple(dot_state_e_c)

print('dot state p:')
print(dot_state_p)

print('dot state e:')
print(dot_state_e)

#%% 球坐标系协态微分方程
spherical_costate_fcn = ldope.spherical_costate_fcn

dot_costate_p_c = costate_type()
dot_costate_e_c = costate_type()

spherical_costate_fcn(state_p0_c, costate_p0_c, control_p0_c, tm_p_c, mu_c,
                      dot_costate_p_c)
spherical_costate_fcn(state_e0_c, costate_e0_c, control_e0_c, tm_e_c, mu_c,
                      dot_costate_e_c)

dot_costate_p = tuple(dot_costate_p_c)
dot_costate_e = tuple(dot_costate_e_c)

print('dot costate p:')
print(dot_costate_p)

print('dot costate e:')
print(dot_costate_e)

#%% 球坐标系控制变量函数
spherical_control_fcn = ldope.spherical_control_fcn

flag_p_c = ctypes.c_int(-1)
flag_e_c = ctypes.c_int(1)

control_p_c = control_type()
control_e_c = control_type()

spherical_control_fcn(state_p0_c, costate_p0_c, flag_p_c, control_p_c)
spherical_control_fcn(state_e0_c, costate_e0_c, flag_e_c, control_e_c)

control_p = tuple(control_p_c)
control_e = tuple(control_e_c)

print('control p:')
print(control_p)

print('control e:')
print(control_e)

#%% 球坐标系哈密顿函数
spherical_hamilton_fcn = ldope.spherical_hamilton_fcn

hamilton_p_c = ctypes.c_double()
hamilton_e_c = ctypes.c_double()

spherical_hamilton_fcn(state_p0_c, costate_p0_c, control_p0_c, tm_p_c, mu_c,
                       ctypes.byref(hamilton_p_c))
spherical_hamilton_fcn(state_e0_c, costate_e0_c, control_e0_c, tm_e_c, mu_c,
                       ctypes.byref(hamilton_e_c))

hamilton_p = hamilton_p_c.value
hamilton_e = hamilton_e_c.value

print('hamilton p:')
print(hamilton_p)

print('hamilton e:')
print(hamilton_e)

#%% 球坐标系扩展状态微分方程
spherical_ext_state_fcn = ldope.spherical_ext_state_fcn

dot_ext_state_c = ext_state_type()

spherical_ext_state_fcn(ext_state_c, tm_p_c, tm_e_c, mu_c, dot_ext_state_c)

dot_ext_state = tuple(dot_ext_state_c)

print('dot ext state:')
print(dot_ext_state)

#%% 球坐标系边界条件
spherical_boundary_fcn = ldope.spherical_boundary_fcn

boundary_c = boundary_type()

spherical_boundary_fcn(ext_state_c, tm_p_c, tm_e_c, mu_c, boundary_c)

boundary = tuple(boundary_c)

print('boundary:')
print(boundary)

#%% 球坐标系适应度函数
spherical_fitness_fcn = ldope.spherical_fitness_fcn

fitness_c = ctypes.c_double()

spherical_fitness_fcn(individual_c, k_c, state_p0_c, state_e0_c, tm_p_c,
                      tm_e_c, du_c, tu_c, ctypes.byref(fitness_c))

fitness = fitness_c.value

print('fitness:')
print(fitness)
