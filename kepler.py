# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 21:59:13 2022

@author: alexh
"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from integrators import *
from simulation_utilities import *
from barnes_hut_algo import octant_node

# magnitude 
def mag(vec):
    summ = 0
    for i in range(len(vec)):
        summ = vec[i]*vec[i] + summ    
    return pow(summ,0.5)

# constant
G = 1
M = 1
m = 1
r1 = 2
v1 = 0.7071
N = 2

# body position
sun_pos = np.array([0,0,0])
sun_rad = 0.03
sun_col = "yellow"

planet_pos = sun_pos + np.array([r1,0,0])
planet_rad = 0.01
planet_col = "blue"

# sun = sphere(sun_pos, sun_rad, sun_col )
# planet = sphere(planet_pos, planet_rad, planet_col)

planet_p = m * np.array([0,v1,0])
planet_p

t = 0
dt = 0.01

# get updated planet position at each time step
new_planet_pos = []
new_planet_vel = []
while t < 3:
    new_planet_pos.append(planet_pos)
    new_planet_vel.append(planet_p / m)
    
    r = planet_pos - sun_pos  # in case the sun moves
    F = (-G*M*m*norm(r))/mag(r)**2    # -G*M*m*norm(r)/mag(r)**2    np.sqrt(r.dot(r))
    planet_p = planet_p + F*dt
    planet_pos = planet_pos + planet_p*dt/m
    
    t += dt

# total body mass    
total_m = np.array([[M],[m]])

# total body velocity    
new_sun_vel = np.tile(sun_pos,(len(new_planet_pos),1)).T
new_planet_vel = np.array(new_planet_vel).T
new_body_vel = np.array([new_sun_vel, new_planet_vel])
# total body position
new_sun_pos = np.tile(sun_pos,(len(new_planet_pos),1)).T    
new_planet_pos = np.array(new_planet_pos).T
new_body_pos = np.array([new_sun_pos, new_planet_pos])

# Make simulation video
# generate_simulation_video(new_body_pos, 20, 2, ['orange','blue'], show_tails=True, file_type="mp4", output_filename="test_2d")
#generate_simulation_video(new_body_pos, 20, 3, ['blue'], show_tails=True, file_type="mp4", output_filename="test_3d")
   
#############

nTime = int(t/dt)

total_energy = []
for i in range(0,nTime):
    input_pos = new_body_pos[:,:,i:i+1].reshape(N,3)
    input_vel = new_body_vel[:,:,i:i+1].reshape(N,3)
    total_energy.append(compute_energy(input_pos, input_vel, total_m, G))
total_energy = np.array(total_energy)

KE, PE = [], []
for i in total_energy:
    KE.append(i[0])
    PE.append(i[1])
KE = np.array(KE)
PE = np.array(PE)

time_axis = np.array(range(nTime))

plot_simulation_energy(time_axis, KE, PE,ax=None,figsize=(8,6))


##############

# Source: https://en.wikipedia.org/wiki/Two-body_problem

m1 = 1 # Mass of particle 1
m2 = 1 # Mass of particle 2
pos1 = np.array([0,0,0]) # Starting position of particle 1
pos2 = np.array([1,0,0]) # Starting position of particle 2

R = (m1*pos1 + m2*pos2)/(m1 + m2) # Find the center of mass as the weighted avg of the starting positions


def x1(t):
    return R + m2/(m1 + m2)*r(t)

def x2(t):
    return R + m1/(m1 + m2)*r(t)


P = 1 # Period
t = 1 # time since perihelion
e = 0 # Assume a circular orbit
a = 1 # Semi-major axis

n = (2*np.pi)/P # Compute the mean motion
M = n*t # Mean anomaly
# Solve Kelpar's equation
# M = E - e *np.sin(E) for E, the eccentric anomoly

#  Then
theta = E = M

r = a*(1 - e*np.cos(E))

a = 1 # The radius of the semi-major axis
mu = G*(m1+m2)
P = 2*np.pi*np.sqrt(a**3 / mu) # https://en.wikipedia.org/wiki/Orbital_period
e = 0 # Assume a circular orbit

# https://en.wikipedia.org/wiki/Kepler%27s_laws_of_planetary_motion#Position_as_a_function_of_time

v = 5
a = 1/(2/r - v**2 /(mu))


#### Initial Conditions ####
G = 1
m1 = 5;m2 = 500
pos_init = np.array([0,2,0]);r = pos_init
vel_init = np.array([0,15.8,0]);v = vel_init
#### Initial Conditions ####

np.sqrt(G*m2/r[1]) # Stable orbit velocity

#### Intermediate Calculations ####
mu = G*(m1+m2) #  standard gravitational parameter
a = 1/(2/r[1] - v[1]**2 /(mu)) # Semi-major axis of orbit

# Solving for eccentricity
h = r.reshape(-1,1) @ (m1*v).reshape(1,-1) # Compute angular momentum
e = np.linalg.norm((v @ h)/mu - r/np.linalg.norm(r)) # https://astronomy.stackexchange.com/questions/29005/calculation-of-eccentricity-of-orbit-from-velocity-and-radius

P = 2*np.pi*np.sqrt(a**3 / mu) # Orbital period https://en.wikipedia.org/wiki/Orbital_period
#### Intermediate Calculations ####

from scipy.optimize import fsolve

#### Function for computing particle location along the elipse at different times ####
def orbit_polar_coords(t):
    n = (2*np.pi)/P # Compute the mean motion
    M = n*t # Mean anomaly
    
    def eccentric_anomaly(E):
        return M - E + np.sin(E)
    
    E = fsolve(eccentric_anomaly,M)[0] # Use an initial guess of M
    
    def true_anomaly(theta):
        return (1-e)*np.tan(theta/2)**2 - (1+e)*np.tan(E/2)**2
    
    theta = fsolve(true_anomaly,M)[0] # Use an initial guess of M
    
    #theta = M
    
    # Compute the heliocentric distance r
    r = a*(1-e*np.cos(E))
    return r, theta

time_steps = np.linspace(0, 10, num=200) # Generate a series of time steps
r_theta_arr = np.array([orbit_polar_coords(t) for t in time_steps]) # Compute the position of the planet in polar coords
x, y = polar_to_xyz(r_theta_arr[:,0],r_theta_arr[:,1]) # Convert back to Cartesian coordinates

pos_agg_planet = np.stack([x,y,np.zeros(len(x))]) # Create an aggregate position array [3 x nT]
pos_agg_sun = np.stack([np.zeros(len(x)),np.zeros(len(x)),np.zeros(len(x))]) # Create an aggregate position array [3 x nT]

pos_agg = np.stack([pos_agg_planet,pos_agg_sun]) # Array dim of (2, 3, nT) to record the positions of both the sun and the
# planet at each time step in 3 dimensions
mass = np.array([m1,m2]) # Create a mass vector

generate_simulation_video(pos_agg, 5, 2, ['orange','blue'], show_tails=True, file_type="mp4", output_filename="test_2d",s=[50,5])





def polar_coords(t):
    """ For circular orbits only"""
    n = (2*np.pi)/P # Compute the mean motion
    M = n*t # Mean anomaly
    theta = E = M
    r = a*(1 - e*np.cos(E))
    return r, theta

time_steps = np.linspace(0, 16, num=200)
r_theta_arr = np.array([polar_coords(t) for t in time_steps])

def polar_to_xyz(r,theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x,y

x, y = polar_to_xyz(r_theta_arr[:,0],r_theta_arr[:,1])

pos_agg_planet = np.stack([x,y,np.zeros(len(x))])
pos_agg_sun = np.stack([np.zeros(len(x)),np.zeros(len(x)),np.zeros(len(x))])

pos_agg = np.stack([pos_agg_planet,pos_agg_sun]) # Array dim of (2, 3, 20)
mass = np.array([m1,m2])

generate_simulation_video(pos_agg, 5, 2, ['orange','blue'], show_tails=True, file_type="mp4", output_filename="test_2d")


def initialize_galaxy(n, total_mass):
    mass = np.random.normal() # Make sure non negative
    pos # Could be normal 
    vel # 
    return pos, vel, mass

