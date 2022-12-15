# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 18:09:55 2022

@author: alexh
"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from integrators import *
from simulation_utilities import *
from kepler_v2 import *
from barnes_hut_algo import octant_node

N = 2 # number of particles 
T = 10
dt = 0.01
G = 1
r = 1
m1 = 50 # Mass of particle 1
m2 = 1 # Mass of particle 2
pos1 = np.array([0,0,0]) # Starting position of particle 1
pos2 = np.array([1,0,0]) # Starting position of particle 2
total_m = np.array([50, 1]).reshape(N,1)
total_pos = np.array([[0,0,0], [1,0,0]])

pos_agg, KE_agg, PE_agg = run_simulation(N=N, T=T, dt=dt, softening=0.1, G=G,
                                         initial_conditions={"mass":np.array([m1, m2]).reshape(N,1),
                                                             "pos":np.array([pos1, pos2]),
                                                             "vel":np.array([0,np.sqrt(m2*G/r) ,0])},
                                         integrator=leap_frog_integrator, use_BH=False, random_state=111)