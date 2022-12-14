# -*- coding: utf-8 -*-
"""
Authors: Eric Helmold, Alex Ho, Ivonne Martinez
AM 111 Final Project: N-body Simulations
"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from integrators import *
from simulation_utilities import *
from barnes_hut_algo import octant_node

N = 2 # number of particles 
m1 = 50 # Mass of particle 1
m2 = 1 # Mass of particle 2
pos1 = np.array([0,0,0]) # Starting position of particle 1
pos2 = np.array([1,0,0]) # Starting position of particle 2
total_m = np.array([m1, m2]).reshape(N,1)
total_pos = np.array([pos1, pos2])

t = 1 # time since perihelion
a = 1 # Semi-major axis
e = 0 # Assume a circular orbit
G = 1

T = 100
dt = 1

def kepler_pos(pos, mass, T, dt, G):
    a = 1
    e = 0
    mu = G*(mass[0]+mass[1])
    per = 2*np.pi*np.sqrt(a**3 / mu)   # https://en.wikipedia.org/wiki/Orbital_period
 
    time_steps = np.linspace(0, T, num=int(T/dt))
    theta_list, r_list = [], []
    ### Calculate polar coordinates 
    n = (2*np.pi)/per # Compute the mean motion
    for t in time_steps:        
        M = n*t # Mean anomaly
        theta = E = M
        r = a*(1 - e*np.cos(E))
        
        theta_list.append(theta)
        r_list.append(r)
    
    x, y = [], []
    x.append(pos[0][0])
    y.append(pos[1][0])
    ### Calculate polar according to x and y
    for i in range(len(theta_list)):
        x.append(r_list[i] * np.cos(theta_list[i]))
        y.append(r_list[i] * np.sin(theta_list[i]))
    
    x = np.array(x).reshape(len(x),)
    y = np.array(y).reshape(len(y),)
    
    pos_agg_planet = np.stack([x,y,np.zeros(len(x))])
    pos_agg_planet.shape
    pos_agg_sun = np.stack([np.zeros(len(x)),np.zeros(len(x)),np.zeros(len(x))])
    
    pos_agg = np.stack([pos_agg_planet,pos_agg_sun]) # Array dim of (2, 3, 20)
    
    return pos_agg 
# epler_pos(total_pos, total_m, T, dt, G).shape

# mass = np.array([m1,m2])
 
# generate_simulation_video(pos_agg, 5, 2, ['blue','orange'], show_tails=True, file_type="mp4", output_filename="test_2d_new")

# Source: https://en.wikipedia.org/wiki/Two-body_problem
