# -*- coding: utf-8 -*-
"""
Authors: Eric Helmold, Alex Ho, Ivonne Martinez
AM 111 Final Project: N-body Simulations
"""

import numpy as np

def leap_frog_integrator(mass, pos, vel, dt, G, softening, acc_calc):
    """
    Updates the position array (pos) for each particle and the velocity array (vel) for each particle
    using the leap-frog integrator method

    Parameters
    ----------
    mass : np.array
        A [N x 1] vector of particle masses        
    pos : np.array
        A [N x 3] matrix of position coordinates (x, y, z)
    vel : np.array
        A [N x 3] matrix of velocities for each particle (vx, vy, vz)
    dt : float
        The size of the time set for each iteration
    G : float
        Newton's Gravitational constant
    softening : float
        An amount of softening used to create spacing between particles to avoid numerical instabilities
    acc_calc : callable
        A callable function used to compute the net acceleration on all n particles in each direction (x,y,z)
    """
    # Compute the acceleration due to gravity in each direction (x,y,z) for each particle
    acc = acc_calc(pos, mass, G, softening)
    
    # First update to the velocities using dt/2 of a sized step
    vel += acc*dt/2.0
    
    # Update the position vectors based on the velocity over 1 dt time step
    pos += vel*dt
    
    # Update the accelerations based on the new positions
    acc = acc_calc(pos, mass, G, softening)
    
    # Second update to the velocities using dt/2 of a sized step
    vel += acc*dt/2.0

def euler_integrator(mass, pos, vel, dt, G, softening, acc_calc):
    """
    Updates the position array (pos) for each particle and the velocity array (vel) for each particle
    using the Euler integration method

    Parameters
    ----------
    mass : np.array
        A [N x 1] vector of particle masses        
    pos : np.array
        A [N x 3] matrix of position coordinates (x, y, z)
    vel : np.array
        A [N x 3] matrix of velocities for each particle (vx, vy, vz)
    dt : float
        The size of the time set for each iteration
    G : float
        Newton's Gravitational constant
    softening : float
        An amount of softening used to create spacing between particles to avoid numerical instabilities
    acc_calc : callable
        A callable function used to compute the net acceleration on all n particles in each direction (x,y,z)
    """
    # Compute the acceleration due to gravity in each direction (x,y,z) for each particle
    acc = acc_calc(pos, mass, G, softening)
    
    # Updates the position vectors first using the starting velocity vectors, updates the velocity vectors using the acc 
    # from the start of the time interval
    pos += vel*dt # Update the position values for 1 step
    vel += acc*dt # Update the velocity values for 1 step
    
def euler_cromer_integrator(mass, pos, vel, dt, G, softening, acc_calc):
    """
    Updates the position array (pos) for each particle and the velocity array (vel) for each particle
    using the Euler-Cromer integration method

    Parameters
    ----------
    mass : np.array
        A [N x 1] vector of particle masses        
    pos : np.array
        A [N x 3] matrix of position coordinates (x, y, z)
    vel : np.array
        A [N x 3] matrix of velocities for each particle (vx, vy, vz)
    dt : float
        The size of the time set for each iteration
    G : float
        Newton's Gravitational constant
    softening : float
        An amount of softening used to create spacing between particles to avoid numerical instabilities
    acc_calc : callable
        A callable function used to compute the net acceleration on all n particles in each direction (x,y,z)
    """
    # Compute the acceleration due to gravity in each direction (x,y,z) for each particle
    acc = acc_calc(pos, mass, G, softening)
    
    # Unlike in the Euler method, here the velocity vectors are updates first, which are then used to update the position
    # vectors using the updated velocities from the end of the time interval
    vel += acc*dt # Update the velocity values for 1 step
    pos += vel*dt # Update the position values for 1 step

def euler_richardson_integrator(mass, pos, vel, dt, G, softening, acc_calc):
    """
    Updates the position array (pos) for each particle and the velocity array (vel) for each particle
    using the Euler-Richardson integration method

    Parameters
    ----------
    mass : np.array
        A [N x 1] vector of particle masses        
    pos : np.array
        A [N x 3] matrix of position coordinates (x, y, z)
    vel : np.array
        A [N x 3] matrix of velocities for each particle (vx, vy, vz)
    dt : float
        The size of the time set for each iteration
    G : float
        Newton's Gravitational constant
    softening : float
        An amount of softening used to create spacing between particles to avoid numerical instabilities
    acc_calc : callable
        A callable function used to compute the net acceleration on all n particles in each direction (x,y,z)
    """
    # Compute the acceleration due to gravity in each direction (x,y,z) for each particle
    acc = acc_calc(pos, mass, G, softening)
    
    # Implements the Euler-Richardson midpoint method
    vel_mid = vel + acc*dt/2.0
    pos_mid = pos + vel*dt/2.0
    acc_mid = acc_calc(pos_mid, mass, G, softening)
    
    # Compute forward step update using the above computed intermediate quantities
    vel += acc_mid*dt
    pos += vel_mid*dt

##########################################################
#### SOMETHING SEEMS TO BE WRONG WITH THIS INTEGRATOR ####
##########################################################
def RK4_integrator(mass, pos, vel, dt, G, softening, acc_calc):
    """
    Updates the position array (pos) for each particle and the velocity array (vel) for each particle
    using the Runge-Kutta 4th order integration method

    Parameters
    ----------
    mass : np.array
        A [N x 1] vector of particle masses        
    pos : np.array
        A [N x 3] matrix of position coordinates (x, y, z)
    vel : np.array
        A [N x 3] matrix of velocities for each particle (vx, vy, vz)
    dt : float
        The size of the time set for each iteration
    G : float
        Newton's Gravitational constant
    softening : float
        An amount of softening used to create spacing between particles to avoid numerical instabilities
    acc_calc : callable
        A callable function used to compute the net acceleration on all n particles in each direction (x,y,z)
    """
    k1r = vel
    k1v = acc_calc(pos, mass, G, softening)
    
    k2r = vel + k1v*(dt/2.0)
    k2v = acc_calc(pos + k1r*(dt/2.0), mass, G, softening)
    
    k3r = vel + k2v*(dt/2.0)
    k3v = acc_calc(pos + k2r*(dt/2.0), mass, G, softening)
    
    k4r = vel + k3v*dt
    k4v = acc_calc(pos + k3r*dt, mass, G, softening)
    
    vel += (dt/6.0) * (k1v + 2*k2v + 2*k3v + k4v)
    pos += (dt/6.0) * (k1r + 2*k2r + 2*k3r + k4r)
    
    
