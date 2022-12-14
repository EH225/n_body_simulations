# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 22:51:55 2022

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
G = 6.67e-11
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

planet_p = m * np.array([0,v1,0])
sun_p = -(planet_p)

t = 0
dt = 0.01

# get updated planet position at each time step
new_planet_pos = []
new_planet_vel = []
while t < 3:
    new_planet_pos.append(planet_pos)
    new_planet_vel.append(planet_p / m)
    
    r = planet_pos - sun_pos  # in case the sun moves
    F = -G*M*m*norm(r)/np.sqrt(r.dot(r))**2    # -G*M*m*norm(r)/mag(r)**2    
   # F = -F
    planet_p = planet_p + F*dt
    sun_p = sun_p + F*dt
    
    planet_pos = planet_pos + planet_p*dt/m
    sun_pos = sun_pos + sun_pos*dt/M
    
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
#generate_simulation_video(new_body_pos, 20, 2, ['orange','blue'], show_tails=True, file_type="mp4", output_filename="test_2d")
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

#plot_simulation_energy(time_axis, KE, PE,ax=None,figsize=(8,6))


####
pi = np.pi  
npoints = 360
r = 1.          # AU
dt = 1./npoints # fractions of a year
mu = 4 * pi**2  # 
x = np.zeros(npoints)
y = np.zeros(npoints)
v_x = np.zeros(npoints)
v_y = np.zeros(npoints)

# Initial Conditions
x[0] = r               # (x0 = r, y0 = 0) AU
v_y[0] = np.sqrt(mu/r) # (v_x0 = 0, v_y0 = sqrt(mu/r)) AU/yr

for step in range(0,npoints-1):
    v_x[step+1]=v_x[step]-4*pi**2*x[step]/(r**3)*dt
    x[step+1]=x[step]+v_x[step+1]*dt
    v_y[step+1]=v_y[step]-4*pi**2*y[step]/(r**3)*dt
    y[step+1]=y[step]+v_y[step+1]*dt

plt.plot(x, y, 'bo')
plt.axis('equal')
plt.show()