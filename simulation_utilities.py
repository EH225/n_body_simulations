# -*- coding: utf-8 -*-
"""
Authors: Eric Helmold, Alex Ho, Ivonne Martinez
AM 111 Final Project: N-body Simulations
"""

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from barnes_hut_algo import octant_node

# Each particle has a 
#   mass (m), 
#   a position vector r = [x,y,z] 
#   and a velocity vector v = [vx, vy, vz]

def compute_acceleration(pos, mass, G, softening=0.1):
    """
    Calculate the acceleration on each particle due to gravity according to Newtonian mechanics

    Parameters
    ----------
    pos : np.array
        A [N x 3] matrix of position coordinates (x, y, z)
    mass : np.array
        A [N x 1] vector of particle masses
    G : float
        Newton's Gravitational constant
    softening : float
        An amount of softening used to create spacing between particles to avoid numerical instabilities

    Returns
    -------
    A [N x 3] np.array containing the acceleration of each particle in the x, y, z directions

    """
    # Positions r = [x,y,z] for all particles, extract vector representations
    x = pos[:,0:1];y = pos[:,1:2];z = pos[:,2:3]

    # Compute all pairwise particle distances along each axis x,y,z, create [N x N] matricies
    # each entry [i,j] is the distance from particle i to particle j in the particular direction computed
    dx = x.T - x # Calculate the distance of each particle from ever other in the x-direction
    dy = y.T - y # Calculate the distance of each particle from ever other in the y-direction
    dz = z.T - z # Calculate the distance of each particle from ever other in the z-direction
    
    # Compute the instantenous acceleration due to gravity on each particle
    inv_r3 = (dx**2 + dy**2 + dz**2 + softening**2)**(-3/2) 
    # Same as 1/((dx^2 + dy^2 + dz^3)^(1/2))^3 = 1/((dx^2 + dy^2 + dz^3)^(3/2)) = (dx^2 + dy^2 + dz^3)^(-3/2)
    np.fill_diagonal(inv_r3,0) # Set the diagonal entries to zero
    
    # F = ma -> a = F/m where F = G m1*m2/r^2 so a = G m1*m2/r^2 / m1 = G m2/r^2
    ax = G*(dx * inv_r3) @ mass # [N x N] @ [N x 1] = [N x 1] acceleration in the x direction for each particle
    ay = G*(dy * inv_r3) @ mass # [N x N] @ [N x 1] = [N x 1] acceleration in the y direction for each particle
    az = G*(dz * inv_r3) @ mass # [N x N] @ [N x 1] = [N x 1] acceleration in the z direction for each particle

    return np.hstack([ax,ay,az]) # Combine all 3 results together and return the combined stacked np.array

def compute_acceleration_BH(pos, mass, G, softening=0.1, theta=0.5, check_tree=False):
    """
    Calculate the acceleration on each particle due to gravity according to Newtonian mechanics using the Barnes Hut algorithm

    Parameters
    ----------
    pos : np.array
        A [N x 3] matrix of position coordinates (x, y, z)
    mass : np.array
        A [N x 1] vector of particle masses
    G : float
        Newton's Gravitational constant
    softening : float
        An amount of softening used to create spacing between very near particles to avoid numerical instabilities
    theta : float
        A parameter in the Barnes Hut algorithm for determining if particles are far enough away to be considered as 1 mass
    check_tree : bool
        A T/F parameter for if optional correctness checks should be run after the construction of the Barnes-Hut tree

    Returns
    -------
    A [N x 3] np.array containing the acceleration of each particle in the x, y, z directions

    """
    # Create a Barnes Hut octant tree for the current particle positions
    x_min, y_min, z_min = pos.min(axis=0) # Find the min for each position dimension (x,y,z)
    x_max, y_max, z_max = pos.max(axis=0) # Find the max for each position dimension (x,y,z)
    # Scale by a factor of 1% so that we get bounds that are slightly larger so that particles are not on the boundaries
    mins = [x_min, y_min, z_min];maxes = [x_max, y_max, z_max] # Collect mins and maxes into lists
    maxes = [bound*1.01 if bound>0 else bound*0.99 for bound in maxes] # Edit the max values to be a little higher
    mins = [bound*0.99 if bound>0 else bound*1.01 for bound in mins] # Edit the min values to be a little lower
    bounds = [item for sublist in zip(mins, maxes) for item in sublist] # Combine and alternate min and max 
        
    BH_tree = octant_node(*bounds) # Unpack the mins and maxes we processed above into the input for the BH tree algo
    
    # Note, we MUST convert this to a (n,) dimensional vector and not use a (n,1) dim vector or else the loop will be passing
    # in mutable elements of the original mass vector which will change the mass of the system as we add particles to the tree
    for particle_pos, particle_mass in zip(pos, mass.copy().reshape(-1)): # Iterate over all the particles in the simulation
        BH_tree.insert(particle_pos,particle_mass) # Add each one to the BH tree using the insert method
    
    if check_tree==True: # If toggled on, then run various checks to make sure the tree is properly constructed
        # Check that the tree contains all the mass and particles we expect it to, these checks are optional
        assert BH_tree.check_structure();assert BH_tree.particle_count == len(pos);assert BH_tree.total_mass == mass.sum()
    
    # Now that we have the BH tree constructed for all particles at their current positions, we can compute the acc on each
    # by generating the relevant set of comparison points using a traversal of the BH tree
    
    # Iteration over all particles in the simulation using list comprehension, compute the subset of particles relevant to the
    # acceleration calculation for each using a traversal of the BH tree and aggregate the accelerations in (x,y,z) into 1 array
    return np.stack([compute_single_particle_acc(particle_pos, *BH_tree.get_traversal(particle_pos, theta),
                                                 G, softening) for particle_pos in pos])
    
def compute_single_particle_acc(particle_pos:np.array, env_pos:np.array, env_mass:np.array, G, softening=0.1)->np.array:
    """
    Computes the net acceleration on a target particle with attributes (particle_pos, particle_mass) vs the other particles 
    in the surrounding environment located at env_pos with masses env_mass

    Parameters
    ----------
    particle_pos : np.array
        An array describing the location in 3d space of the target particle (x,y,z), note, the target particle mass is not 
        needed since we are computing acceleration so it will cancel out in our calculation
    env_pos : np.array
        An array describing the locations of other bodies of mass in 3d space (x,y,z) of size [n x 3]
    env_mass : np.array
        An array describing the masses of other bodies of mass in 3d space of size [n]
    G : float
        Newton's Gravitational constant
    softening : float
        An amount of softening used to create spacing between very near particles to avoid numerical instabilities

    Returns
    -------
    A 3 dimensional array detailing the net acceleration on the candidate particle due to gravity from the other bodies of
    mass in the environment surrounding it
    """
    if type(env_pos)==list:
        env_pos = np.stack(env_pos)
    if type(env_mass)==list:
        env_mass = np.stack(env_mass)
    
    # Compute the coordinate-wise distance between the target particle and each body of mass in the environment
    # Bodies of mass may be individual particles or may be a collection of particles according to the BH algorithm    
    coord_dist = env_pos - particle_pos # Take x_env - x_tgt for each particle in the environment [n x 3] - [1 x 3] -> [n x 3]
    inv_r3 = ((coord_dist**2).sum(axis=1) + softening**2)**(-3/2) # Calculate the denominator factor [n x 1]
    
    # F = ma -> a = F/m where F = G m1*m2/r^2 so a = G m1*m2/r^2 / m1 = G m2/r^2
    # Compute the instantenous acceleration due to gravity on each particle
    return env_mass.T @ (coord_dist * inv_r3.reshape(-1,1))*G
    
def compute_acceleration_BH_closure(theta=0.5, check_tree=False):
    
    def compute_acceleration_BH_red(pos, mass, G, softening=0.1):
        """
        Calculate the acceleration on each particle due to gravity according to Newtonian mechanics using the Barnes Hut algorithm
    
        Parameters
        ----------
        pos : np.array
            A [N x 3] matrix of position coordinates (x, y, z)
        mass : np.array
            A [N x 1] vector of particle masses
        G : float
            Newton's Gravitational constant
        softening : float
            An amount of softening used to create spacing between very near particles to avoid numerical instabilities
        theta : float
            A parameter in the Barnes Hut algorithm for determining if particles are far enough away to be considered as 1 mass
        check_tree : bool
            A T/F parameter for if optional correctness checks should be run after the construction of the Barnes-Hut tree
    
        Returns
        -------
        A [N x 3] np.array containing the acceleration of each particle in the x, y, z directions
    
        """
        # Create a Barnes Hut octant tree for the current particle positions
        x_min, y_min, z_min = pos.min(axis=0) # Find the min for each position dimension (x,y,z)
        x_max, y_max, z_max = pos.max(axis=0) # Find the max for each position dimension (x,y,z)
        # Scale by a factor of 1% so that we get bounds that are slightly larger so that particles are not on the boundaries
        mins = [x_min, y_min, z_min];maxes = [x_max, y_max, z_max] # Collect mins and maxes into lists
        maxes = [bound*1.01 if bound>0 else bound*0.99 for bound in maxes] # Edit the max values to be a little higher
        mins = [bound*0.99 if bound>0 else bound*1.01 for bound in mins] # Edit the min values to be a little lower
        bounds = [item for sublist in zip(mins, maxes) for item in sublist] # Combine and alternate min and max 
            
        BH_tree = octant_node(*bounds) # Unpack the mins and maxes we processed above into the input for the BH tree algo
        
        # Note, we MUST convert this to a (n,) dimensional vector and not use a (n,1) dim vector or else the loop will be passing
        # in mutable elements of the original mass vector which will change the mass of the system as we add particles to the tree
        for particle_pos, particle_mass in zip(pos, mass.copy().reshape(-1)): # Iterate over all the particles in the simulation
            BH_tree.insert(particle_pos,particle_mass) # Add each one to the BH tree using the insert method
        
        if check_tree==True: # If toggled on, then run various checks to make sure the tree is properly constructed
            # Check that the tree contains all the mass and particles we expect it to, these checks are optional
            assert BH_tree.check_structure();assert BH_tree.particle_count == len(pos);assert BH_tree.total_mass == mass.sum()
        
        # Now that we have the BH tree constructed for all particles at their current positions, we can compute the acc on each
        # by generating the relevant set of comparison points using a traversal of the BH tree
        
        # Iteration over all particles in the simulation using list comprehension, compute the subset of particles relevant to the
        # acceleration calculation for each using a traversal of the BH tree and aggregate the accelerations in (x,y,z) into 1 array
        return np.stack([compute_single_particle_acc(particle_pos, *BH_tree.get_traversal(particle_pos, theta),
                                                     G, softening) for particle_pos in pos])
    
    # Return the reduced (red) version of the compute_acceleration_BH function with theta and check_tree bound to it
    return compute_acceleration_BH_red

def compute_energy(pos, vel, mass, G):
    """
    Calculate the kinetic energy (KE) and potential energy (PE) of the system of particles

    Parameters
    ----------
    pos : np.array
        A [N x 3] matrix of position coordinates (x, y, z)
    vel : np.array
        A [N x 3] matrix of velocities for each particle (vx, vy, vz)
    mass : np.array
        A [N x 1] vector of particle masses
    G : float
        Newton's Gravitational constant

    Returns
    -------
    KE : float
        The total kinetic energy across all particles in the system
    PE : float
        The total potential energy across all particles in the system

    """
    ### Kinetic Energy ###
    # Compute kinetic energy where KE = (1/2)mv^2 with m = particle mass, v = particle velocity
    KE = (0.5)*((vel**2).sum(axis=1) @ mass)[0]
    
	### Potential Energy ###

    # Positions r = [x,y,z] for all particles, extract vector representations
    x = pos[:,0:1];y = pos[:,1:2];z = pos[:,2:3]

    # Compute all pairwise particle distances along each axis x,y,z, create [N x N] matricies
    # each entry [i,j] is the distance from particle i to particle j in the particular direction computed
    dx = x.T - x # Calculate the distance of each particle from ever other in the x-direction
    dy = y.T - y # Calculate the distance of each particle from ever other in the y-direction
    dz = z.T - z # Calculate the distance of each particle from ever other in the z-direction

	# matrix that stores 1/r for all particle pairwise particle separations 
    inv_r = np.sqrt(dx**2 + dy**2 + dz**2) # Compute the L2-norm Euclidian distance between each object
    inv_r[inv_r>0] = 1.0/inv_r[inv_r>0] # Invert as 1/R where R = distance between particles for distances > 0
    # This gives us a symetric [N x N] matrix of 1/R values for the distance from the ith particle to the jth particle
    
    
    # PE = G*m1*m2/r where G is a constant, m1 and m2 are the masses of each particle pair and r is the distance between them
    PE = G * np.triu(-(mass @ mass.T)*inv_r,1).sum() # Use an upper triangular matrix of mass @ mass.T to only count each
    # pairwise interaction term once so that we do not double count m1 to m2 and m2 to m1.
    
    return (KE, PE)


def run_simulation(N, T, dt, softening, G, integrator, normalize_momentum=True, initial_conditions=None, use_BH=False, theta=0.5, random_state=111):
    """
    Main wrapper function for running the N-body simulation for a given set of input parameters

    Parameters
    ----------
    N : int
        Specifies the number of particles in the simulation
    T : int
        Specifies the total run time of the simulation (from 0 to T)
    dt : float
        Specifies the time step incriment over which the simulation will compute updates
    softening : float
        An amount of softening used to create spacing between particles to avoid numerical instabilities
    G : float
        Newton's Gravitational constant
    integrator : callable function
        A time-step integrator capable of updating the position and velocity arrays
    normalize_momentum : bool
        A toggle for normalizing the total momentum of the system to be net zero in all 3 directions for stability
    initial_conditions : dict
        A dictionary of initial conditions, if provided will override the random initialization of initial conditions
        Must provide as keys: mass, pos, vel with their associated np.arrays
    use_BH : bool
        A toggle to select if the Barnes-Hut algorithm should be used in the computation of acceleration in this simulation
    theta : float
        A parameter in the Barnes Hut algorithm for determining if particles are far enough away to be considered as 1 mass
    random_state : int
        A random state for replicating the random initialization of the initial conditions

    Returns
    -------
    pos_agg : np.array
        A [N x 3 x T/dt] array holding the (x,y,z) position of each of the N particles for each time step
    KE_agg : np.array
        A [T/dt x 1] array holding the kinetic energy of the system at each time step
    PE_agg : np.array
        A [T/dt x 1] array holding the potential energy of the system at each time step

    """
    ###
    ### Initial set up for simulation ###
    ###
    np.random.seed(random_state) # Set a random seed for replicating results
    #t = 0.0 # Current time of the simulation, initialize at 0 (will incriment up to T)
    if initial_conditions is None: # If not provided, then randomly initialize
        mass = 20.0*np.ones((N,1))/N  # Total mass of all particles together is 20, evenly split
        pos  = np.random.randn(N,3)   # Randomly generate initial positions using N(0,1)
        vel  = np.random.randn(N,3)   # Randomly generate initial velocities using N(0,1)
    else: # Otherwise, use the initial conditions passed in
        mass = initial_conditions["mass"];pos = initial_conditions["pos"];vel = initial_conditions["vel"]
        assert len(mass) == pos.shape[0], f"mass and pos initial conditions must have the same row length, recieved {mass.shape} and {pos.shape}"
        assert pos.shape == vel.shape, f"pos and vel initial conditions must have the same shape, recieved {pos.shape} and {vel.shape}"
        N = len(mass) # Set the number of particles
        
    Nt = int(np.ceil(T/dt)) # Calculate the number of total timestep iterations
    
    # Alter the starting velocities so that the total momentum of the system is zero in the x,y,z directions
    # so that the system as a whole does not drift off screen during the simulation, convert to Center-of-Mass frame
    if normalize_momentum==True:
        vel -= np.mean(mass * vel, axis=0) / np.mean(mass)
    
    KE, PE  = compute_energy(pos, vel, mass, G) # Calculate initial energy of system, kinetic and potential
    
    # Create data-aggregation objects to store the values of our simulation at each time step
    pos_agg = np.zeros((N,3,Nt+1));pos_agg[:,:,0] = pos # Create a [N x 3 x Nt+1] array to store particle positions
    KE_agg = np.zeros(Nt+1);KE_agg[0] = KE # Create a [Nt+1] array to store the system KE
    PE_agg = np.zeros(Nt+1);PE_agg[0] = PE # Create a [Nt+1] array to store the system PE    
    
    if use_BH==True: # If set to True, then use the Barnes-Hut algorithm to calculate acceleration
        acc_calc = compute_acceleration_BH_closure(theta) # Use a closure to bind theta to the function
    else: # Otherwise, calculate acceleration the standard way using the full particle-to-particle interationcs
        acc_calc = compute_acceleration
    
    ###
    ### Run the simulation until T ###
    ###
        
    for i in tqdm(range(Nt)): # Loop through each time step
        # Update the postion vector and velocity vector for each particle using the integrator
        integrator(mass, pos, vel, dt, G, softening, acc_calc)
        
        # Archive new particle positions and system energy measures
        pos_agg[:,:,i+1] = pos
        KE_agg[i+1], PE_agg[i+1] = compute_energy(pos, vel, mass, G)
    
    return pos_agg, KE_agg, PE_agg


def plot_simulation_energy(time_axis,KE_agg,PE_agg,ax=None,figsize=(8,6)):
    if ax is None:
        fig,ax = plt.subplots(1,1,figsize=figsize)
    ax.plot(time_axis,KE_agg, label="Kinetic Energy",zorder=3)
    ax.plot(time_axis,PE_agg, label="Potential Energy",zorder=3)
    ax.plot(time_axis,KE_agg+PE_agg,label="Total Energy",zorder=3)
    ax.legend();ax.grid(color='lightgray',zorder=-3)
    ax.set_xlabel("Time");ax.set_ylabel("Energy")
    ax.set_title("Kinetic, Potential and Total Energy of the System over Time")
    


# Importing movie py libraries
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
# https://www.geeksforgeeks.org/moviepy-creating-animation-using-matplotlib/
# https://zulko.github.io/moviepy/getting_started/videoclips.html


def generate_simulation_video(pos_agg, T, dim_size=3, color_scheme=["blue"], show_tails=True,
                              tail_len=50, figsize=(10,10), xlim=(-3, 3), ylim=(-3, 3), zlim=(-3, 3), legend=None,
                              s=np.array([10]), file_type='mp4', output_filename="my_vid"):
    """
    Generates simulation videos as either an mp4 of gif. Capable of generating simulations in 3d or 2d with flexible plotting parameters

    Parameters
    ----------
    pos_agg : np.array
        An array containing the position data for each particle in the simulation at each time step of size [n x 3 x Nt] where n is the 
        number of particles in the simulation, 3 represents the 3 spatial dimentions (x,y,z) and Nt is the total number of time step
        iterations used in the simulation
    T : float
        The length of the output simulation video in seconds
    dim_size : int, optional
        To specify either 2 or 3 dimensional video output. The default is 3.
    color_scheme : list, optional
        A list containing either 1 color for all particles or a separate color for each particle in the simulation
    show_tails : bool, optional
        A toggle that when set to True will also plot the trailing positions of each particle in time in a ligher color. The default is True.
    tail_len : int, optional
        The number of trailing positions to plot if show_tails == True. The default is 50.
    figsize : tuple, optional
        A length 2 tuple describing the plot size. The default is (10,10).
    xlim : tuple, optional
        A length 2 tuple describing the x-axis limits. The default is (-3, 3).
    ylim : tuple, optional
        A length 2 tuple describing the y-axis limits. The default is (-3, 3).
    zlim : tuple, optional
        A length 2 tuple describing the z-axis limits. The default is (-3, 3).
    legend : list
        Optional list that can be passed into label frames
    s : np.array, optional
        A list containing either 1 size for all particles or a separate size for each particle in the simulation. The default is [10].
    file_type : str, optional
        Specification for the file type that the video should be saved as, either mp4 (recommended) or gif. The default is 'mp4'.
    output_filename : str, optional
        The name of the output file, not including the file extension which is specified in file_type. The default is "my_vid".
    """
    file_types=["gif","mp4"];dim_sizes = [2,3]
    assert file_type in file_types, f"file_type must be one of the following: {file_types}"
    assert dim_size in dim_sizes, f"dim_size must be one of the following: {dim_sizes}"
    
    fig, ax = plt.subplots(figsize=figsize) # Create a plotting space
    ax.axis('off') # Remove x and y axis labels on outer plot area and outer box
    fps=int(pos_agg.shape[2]/T) # Set the FPS to be the number of position vector obs / T so that the overall clip length matches T
    N = pos_agg.shape[0] # The particle count
    
    if dim_size == 3:    
        ax = fig.add_subplot(projection='3d') # Create 3d projection sub-axis
    
    if len(s) == 1: # If we have been given a length 1 list
        s = s[0]  # Then extract that element as the size for all particles
        multi_size=False
        frame_tail_s_global = s
    else:
        assert pos_agg.shape[0] == len(s), f"s must be the same length as n, the number of particles, got {len(s)} and {pos_agg.shape[0]}"
        multi_size=True
        frame_tail_s_global = np.repeat(s, tail_len + 1, axis=0)/20
    
    if len(color_scheme) == 1: # If only 1 color is passed in
        multi_color=False
        if color_scheme=="red":
            main_col = [1,0,0];tail_col = [1.0,0.7,0.7]
        elif color_scheme=="green":
            main_col = [0,1,0];tail_col = [0.7,1.0,0.7]
        else:
            main_col = [0,0,1];tail_col = [0.7,0.7,1.0]
        frame_tail_col_global = tail_col
    else: # If multiple colors are passed in, then perform dimensionality checks and convert all
        assert len(color_scheme) == pos_agg.shape[0], f"color_scheme and pos_agg number of particles mismatch: {len(color_scheme)} and {pos_agg.shape}"
        main_col_dict = {"red":[1,0,0],"green":[0,1,0],"blue":[0,0,1]}
        tail_col_dict = {"red":[1,0.7,0.7],"green":[0.7,1,0.7],"blue":[0.7,0.7,1]}
        main_col = np.array([main_col_dict[col] for col in color_scheme])
        tail_col = np.array([tail_col_dict[col] for col in color_scheme])
        multi_color=True
        frame_tail_col_global = np.repeat(tail_col, tail_len + 1, axis=0)
        
    def make_frame_2d(t):
        t = int(t*fps) # Convert to int for indexing purposes
        ax.clear() # Clear the plot for the next image to be added
        ax.axis('off') # Remove x and y axis labels on outer plot area and outer box
        
        if show_tails:
            xx = pos_agg[:,0,max(t-tail_len,0):t+1].reshape(-1) # Get the trailing 50 position obs in the x direction
            yy = pos_agg[:,1,max(t-tail_len,0):t+1].reshape(-1) # Get the trailing 50 position obs in the y direction
            
            if t <= tail_len: # Check if we are below the tail_len, if so, then we need to make custom len vector expansions 
                if multi_color==True: # If there are multiple colors (one for each particle)
                    frame_tail_col = np.repeat(tail_col, xx.shape[0]//N, axis=0) # Then expand the color vector accordingly
                else: # Otherwise, use the tail color that is common to all particles if only 1 color specified
                    frame_tail_col = tail_col
                
                if multi_size==True: # If there are multiple particle sizes (one for each particle)
                    frame_tail_s = np.repeat(s, xx.shape[0]//N, axis=0)/20 # Then expand the size vector accordingly
                else: # Otherwise, use the dot size  that is common to all particle tails if only 1 specified
                    frame_tail_s = s/20
            
                ax.scatter(xx,yy,s=frame_tail_s,color=frame_tail_col) # Plot the particle tails 
            
            else: # If we are beyond the initial period, all tail lengths will be equal to tail_len so we can use the same s and col
                ax.scatter(xx,yy,s=frame_tail_s_global,color=frame_tail_col_global) # Plot the particle tails 
                                
            
        ax.scatter(pos_agg[:,0,t],pos_agg[:,1,t],s=s,color=main_col) # Plot the most recent obs in dark blue
        ax.set(xlim=xlim, ylim=ylim) # Set x and y axis limits
        #ax.set_aspect('equal', 'box')
        ax.set_xticks([]);ax.set_yticks([]) # Remove x and y axis ticks
        return mplfig_to_npimage(fig) # Return numpy image
       
    def make_frame_3d(t):
        t = int(t*fps) # Convert to int for indexing purposes
        ax.clear() # Clear the plot for the next image to be added
        
        if show_tails:
            xx = pos_agg[:,0,max(t-tail_len,0):t+1].reshape(-1) # Get the trailing position obs in the x direction
            yy = pos_agg[:,1,max(t-tail_len,0):t+1].reshape(-1) # Get the trailing position obs in the y direction
            zz = pos_agg[:,2,max(t-tail_len,0):t+1].reshape(-1) # Get the trailing position obs in the y direction   
            
            if t <= tail_len: # Check if we are below the tail_len, if so, then we need to make custom len vector expansions 
                if multi_color==True: # If there are multiple colors (one for each particle)
                    frame_tail_col = np.repeat(tail_col, xx.shape[0]//N, axis=0) # Then expand the color vector accordingly
                else: # Otherwise, use the tail color that is common to all particles if only 1 color specified
                    frame_tail_col = tail_col
                
                if multi_size==True: # If there are multiple particle sizes (one for each particle)
                    frame_tail_s = np.repeat(s, xx.shape[0]//N, axis=0)/20 # Then expand the size vector accordingly
                else: # Otherwise, use the dot size  that is common to all particle tails if only 1 specified
                    frame_tail_s = s/20
                
                ax.scatter(xx,yy,zz,s=frame_tail_s,color=frame_tail_col,zorder=1) # Plot the trailing obs in light color
            
            else: # If we are beyond the initial period, all tail lengths will be equal to tail_len so we can use the same s and col
                ax.scatter(xx,yy,zz,s=frame_tail_s_global,color=frame_tail_col_global,zorder=1) # Plot the trailing obs in light color
        
        ax.scatter(pos_agg[:,0,t],pos_agg[:,1,t],pos_agg[:,2,t]
                   ,s=s,color=main_col,zorder=3) # Plot the most recent obs in dark color
         
        ax.set(xlim=xlim, ylim=ylim, zlim=zlim) # Set x, y, z axis limits
        if legend is not None:
            ax.legend(handles=legend)
        #ax.set_aspect('equal', 'box')
        #ax.set_xticks([]);ax.set_yticks([]) # Remove x and y axis ticks
        return mplfig_to_npimage(fig) # Return numpy image
    
    # Create an animation object to be used in the animation generation
    if dim_size == 2:
        animation = VideoClip(make_frame_2d, duration = T)
    elif dim_size == 3:
        animation = VideoClip(make_frame_3d, duration = T)
    else:
        raise ValueError(f"dim_size must be one of the following: {dim_sizes}")
    
    if file_type == "gif":
        animation.write_gif(output_filename+"."+file_type,fps=fps,program= 'imageio')
    elif file_type == "mp4":
        animation.write_videofile(output_filename+"."+file_type,fps=fps)
    else:
        raise ValueError(f"file_type must be one of the following: {file_types}")