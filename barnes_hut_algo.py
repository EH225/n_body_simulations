# -*- coding: utf-8 -*-
"""
Authors: Eric Helmold, Alex Ho, Ivonne Martinez
AM 111 Final Project: N-body Simulations
Source:https://www.cs.princeton.edu/courses/archive/fall03/cs126/assignments/barnes-hut.html
Source: http://arborjs.org/docs/barnes-hut
Source: https://beltoforion.de/en/barnes-hut-galaxy-simulator/
"""

### Package Imports ###
import numpy as np
import matplotlib.pyplot as plt
### Package Imports ###

###################################### ###################################### ######################################
###################################### ### The Barnes-Hut Algorithm in 2d ### ######################################
###################################### ###################################### ######################################

class quadrant_node:
    def __init__(self,x_min:float,x_max:float,y_min:float,y_max:float):
        """Constructor for each quadrant node object"""
        # Attributes describing the bounds of this quadrant sub-divisions
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        
        self.total_mass = 0 # Records the sum of all mass for each particle in this quadrant
        self.particle_count = 0 # Records how many particles are in this quadrant

    def contains(self,pos_vec)->bool:
        """Returns T/F indicating if a given (x,y) coordinate pairing is contained within the quadrant"""
        x,y = pos_vec # Unpack the coordinates of the input position vector into x and y components
        return x >= self.x_min and x < self.x_max and y >= self.y_min and y <= self.y_max
    
    def insert(self, pos_vec:np.ndarray, mass_val:float):
        """Method for adding a new particle to the BH tree, recursively creates new quadrants"""
        if self.particle_count == 0 and self.contains(pos_vec): # Check if the particle is cotained in this node
            self.center_of_mass = pos_vec # Record the position of the particle as the center of mass
            self.total_mass = mass_val # Record the mass of the particle
            self.particle_count += 1 # Incriment up the particle count for this node (and all of its children) 
        elif self.particle_count == 1: # If the particle count is equal to 1, then we have to convert this node from an
            # external or leaf node into an internal node (i.e. one with child nodes). First, create a series of child
            # nodes for a new quadrant sub-division and pass down the existing particle into the appropriate quadrant
            # Create new child nodes in the tree using the quadrant_node class and create smaller boundary divisions
            x_mid, y_mid = (self.x_min+self.x_max)/2, (self.y_min+self.y_max)/2 # Compute the midpoints of the current rectangle
            self.child_NW = quadrant_node(self.x_min,x_mid,y_mid,self.y_max)
            self.child_NE = quadrant_node(x_mid,self.x_max,y_mid,self.y_max)
            self.child_SE = quadrant_node(x_mid,self.x_max,self.y_min,y_mid)
            self.child_SW = quadrant_node(self.x_min,x_mid,self.y_min,y_mid)
            # Next, find which new sub-node quadrant the current particle stored in this node can be passed to
            self._pass_to_child(self.center_of_mass, self.total_mass)
            # Next update the center-of-mass and total mass at this node
            self.center_of_mass = (self.center_of_mass*self.total_mass + pos_vec*mass_val) / (self.total_mass + mass_val)
            self. total_mass += mass_val
            self.particle_count +=1
            # Now pass down the new particle down to a child sub-quadrant
            self._pass_to_child(pos_vec, mass_val)
        else: # Otherwise, if this node already has particles in it, then update the center-of-mass and total mass at this node
            self.center_of_mass = (self.center_of_mass*self.total_mass + pos_vec*mass_val) / (self.total_mass + mass_val)
            self.total_mass += mass_val # Incriment up the total mass of this node based on the new point added 
            self.particle_count += 1 # Record the number of points within this quadrant
            # At this point with 2 or more particles, there will be child sub-quadrants, pass the new particle into the next layer
            self._pass_to_child(pos_vec, mass_val)

    def _pass_to_child(self, pos_vec:np.ndarray, mass_val:float):
        """Internal helper method for passing a new particle down into a node's child sub-quadrants"""
        if self.child_NW.contains(pos_vec): # Check for each quadrant where this new particle belongs
            self.child_NW.insert(pos_vec,mass_val)
        elif self.child_NE.contains(pos_vec):
            self.child_NE.insert(pos_vec,mass_val)
        elif self.child_SE.contains(pos_vec):
            self.child_SE.insert(pos_vec,mass_val)
        elif self.child_SW.contains(pos_vec):
            self.child_SW.insert(pos_vec,mass_val)
        else:
            raise AttributeError("No suitable sub-quadrant found for new particle:",pos_vec)
    
    def plot_boundaries(self,ax=None,col:str='black',linewidth=0.5):
        """A recursive method for plotting the quadrant sub-divisions made by the algorithm"""
        if ax is None:
            fig,ax = plt.subplots(1,1)
    
        if self.particle_count > 1: # If there are other particles, then recursively call plot_boundaries on child nodes
            self.child_NW.plot_boundaries(ax,col,linewidth)
            self.child_NE.plot_boundaries(ax,col,linewidth)
            self.child_SE.plot_boundaries(ax,col,linewidth)
            self.child_SW.plot_boundaries(ax,col,linewidth)
        
        else: # If no child nodes, then plot the boundaries of this node
            ax.plot([self.x_min,self.x_max],[self.y_min,self.y_min],color=col,linewidth=linewidth) # Add the bottom border
            ax.plot([self.x_min,self.x_max],[self.y_max,self.y_max],color=col,linewidth=linewidth) # Add the upper border
            ax.plot([self.x_min,self.x_min],[self.y_min,self.y_max],color=col,linewidth=linewidth) # Add the left border
            ax.plot([self.x_max,self.x_max],[self.y_min,self.y_max],color=col,linewidth=linewidth) # Add the right border
            
    def check_structure(self):
        """Helper method for checking that the tree has been properly constructed"""
        assert self.total_mass == self._get_descendent_mass() # Check that the total mass matches the sum of the descendants' mass
        assert self.particle_count == self._get_descendant_particles() # Check that the total particle count matches
        # the sum of particles across all of the descendants
        print("All checks passed!")
        return True
    
    def _get_descendent_mass(self):
        """Helper function for check_structure to gather descendant node masses"""
        if self.particle_count<=1: # If 1 or 0 particles in this quadrant, then return the total_mass of the node
            return self.total_mass
        else: # Otherwise return the sum overall the descendant nodes
            return self.child_NE._get_descendent_mass() + self.child_NW._get_descendent_mass() + \
                self.child_SE._get_descendent_mass() + self.child_SW._get_descendent_mass()
    
    def _get_descendant_particles(self): 
        """Helper function for check_structure to gather descendant node particles"""
        if self.particle_count<=1: # If 1 or 0 particles in this quadrant, then return the particle_count of the node
            return self.particle_count
        else: # Otherwise return the sum overall the descendant nodes
            return self.child_NE._get_descendant_particles() + self.child_NW._get_descendant_particles() + \
                self.child_SE._get_descendant_particles() + self.child_SW._get_descendant_particles()
    
    def get_traversal(self, pos_vec:np.ndarray, theta:float=0.5, assert_check=False):
        """Returns a pos np.array and a mass np.array containing the positions (x,y) and a masses of a set of particles
           to be used in the net acceleration calculation based on the Barnes-Hut proximity rule that treates distant
           masses and 1 combined center of mass. Theta controls the extent to which we summarize particles as 1 mass.
           If s (the width of the node's region) divided by d (the distance to the center of mass of that node) is
           less than theta, then the internal node is sufficiently far away to treat as 1 center-of-mass. If theta==0,
           then we will never perform any summarizing, this method will return the positions of all particles. If theta
           is very large, then this method will return just 1 particle i.e. all other summarized as 1 node. 
        """       
        if self.particle_count == 1: # If there is only 1 particle in the node quadrant, return it
            return [self.center_of_mass], [self.total_mass]
        
        elif self.particle_count > 1: # If there is more than just 1 particle, determine if we should treat them as 1 mass
            s = ((self.x_max - self.x_min) + (self.y_max - self.y_min))/2 # Compute a width for this quadrant as the average
            # width along the x and y dimensions since sub-quadrants may not necessarily be square
            # Compute the distance from the input pos_vector to the node's center-of-mass
            d = self.euclidian_dist(pos_vec,self.center_of_mass)
            if s / d < theta: # If the quadrant is sufficiently small relative to the distance away, treat it as 1 combined mass
                return [self.center_of_mass], [self.total_mass]
            
            else: # Otherwise, recursively traverse the tree to build a set of position vectors and mass values
                pos_agg = [];mass_agg=[] # Create blank lists to hold the outputs from the recursive calls
                if self.child_NW.particle_count>0: # Check if the child node is non-empty
                    pos_list, mass_list = self.child_NW.get_traversal(pos_vec, theta) # Get relevant points from sub-quadrant
                    pos_agg+=pos_list;mass_agg+=mass_list # Append output from recursive traversal to aggregation lists
                
                if self.child_NE.particle_count>0: # Check if the child node is non-empty
                    pos_list, mass_list = self.child_NE.get_traversal(pos_vec, theta) # Get relevant points from sub-quadrant
                    pos_agg+=pos_list;mass_agg+=mass_list # Append output from recursive traversal to aggregation lists
                
                if self.child_SE.particle_count>0: # Check if the child node is non-empty
                    pos_list, mass_list = self.child_SE.get_traversal(pos_vec, theta) # Get relevant points from sub-quadrant
                    pos_agg+=pos_list;mass_agg+=mass_list # Append output from recursive traversal to aggregation lists
                
                if self.child_SW.particle_count>0: # Check if the child node is non-empty
                    pos_list, mass_list = self.child_SW.get_traversal(pos_vec, theta) # Get relevant points from sub-quadrant
                    pos_agg+=pos_list;mass_agg+=mass_list # Append output from recursive traversal to aggregation lists
                
                if assert_check:
                    assert len(pos_agg) == len(mass_agg)
                return pos_agg, mass_agg
        
    def euclidian_dist(self, pos_vec1, pos_vec2):
        """Computes the Euclidian distance between 2 position vectors"""
        return np.linalg.norm(pos_vec1 - pos_vec2)
    
    def __str__(self):
        """A string representation method used to print out the tree structure when called through print"""
        if self.particle_count==0:
            return "-NA"
        elif self.particle_count==1:
            return f'BH-Tree({self.particle_count}, {self.total_mass})'
        else:
            return f'BH-Tree({self.particle_count}, {self.total_mass})' + \
                   '\n|\n|-(NW)->' + '\n|      '.join(str(self.child_NW).split('\n')) + \
                   '\n|\n|-(NE)->' + '\n|      '.join(str(self.child_NE).split('\n')) + \
                   '\n|\n|-(SE)->' + '\n|      '.join(str(self.child_SE).split('\n')) + \
                   '\n|\n|-(SW)->' + '\n|      '.join(str(self.child_SW).split('\n'))


#############################
### Testing the algorithm ###
#############################

if __name__ == "__main__":
    # Generate a sample data set
    n = 30 # The number of particles to simulate
    pos = np.random.random(n*2).reshape(n,2) # A n x 2 vector of (x,y) position coordinates for n particles
    mass = np.ones(len(pos)).reshape(-1,1) # Give each particle a mass of 1
    
    BH_tree = quadrant_node(0,1,0,1) # Initialize a BH tree from x:[0,1] to y:[0:1], make sure to initialize so
    # that all of the particles are contained within the initial node space, otherwise we will have an error
    # try to round up to the next nearest integer for that to happen
    
    for particle_pos,particle_mass in zip(pos, mass.reshape(-1)): # Iterate over all the particles
        BH_tree.insert(particle_pos,particle_mass) # Add each one to the BH tree
    
    # Check that the tree contains all the mass and particles we expect it to
    assert BH_tree.check_structure();assert BH_tree.particle_count == len(pos);assert BH_tree.total_mass == mass.sum()
    print(BH_tree) # Print the string representation of the Barnes Hut Tree
    
    # Plot the data points along with the boundaries in 2D
    fig,axes = plt.subplots(1,1,figsize=(6,6))
    axes.scatter(pos[:,0],pos[:,1],s=20);axes.set_xlabel("X");axes.set_ylabel("Y")
    axes.set_title("Barnes-Hut Tree Sub-Set Bounaries")
    BH_tree.plot_boundaries(axes) # Add Barnes-Hut algorithm bounary lines
    
    # Testing the tree-traversal
    test_point = np.array([0.5,0.5]);theta=1.50 # Generate a test point and a theta value
    pos_agg, mass_agg = BH_tree.get_traversal(test_point, theta)
    
    pos_agg = np.stack(pos_agg);mass_agg = np.array(mass_agg) # Convert to np.arrays
    print("pos_agg.shape",pos_agg.shape);print("mass_agg.shape",mass_agg.shape)
    
    # Plot a visualization of the original data set and the Barnes-Hut reduced data set
    fig,axes = plt.subplots(1,1,figsize=(6,6))
    axes.scatter(pos[:,0],pos[:,1],label="Original");axes.set_xlabel("X");axes.set_ylabel("Y")
    axes.scatter(pos_agg[:,0],pos_agg[:,1],label="BH Reduction",marker="^")
    axes.scatter(test_point[0],test_point[1],marker=",",label="Eval Point");axes.legend()
    axes.set_title("Barnes-Hut Point Reduction - Theta="+str(theta))
    #BH_tree.plot_boundaries(axes) # Add Barnes-Hut algorithm bounary lines
    plt.show()



###################################### ###################################### ######################################
###################################### ### The Barnes-Hut Algorithm in 3d ### ######################################
###################################### ###################################### ######################################

class octant_node:
    def __init__(self,x_min:float,x_max:float,y_min:float,y_max:float,z_min:float,z_max:float):
        """Constructor for each octant node object"""
        # Attributes describing the bounds of this octant sub-division
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max
        
        self.total_mass = 0 # Records the sum of all mass for each particle in this octant
        self.particle_count = 0 # Records how many particles are in this octant

    def contains(self,pos_vec)->bool:
        """Returns T/F indicating if a given (x,y,z) coordinate pairing is contained within the octant"""
        x,y,z = pos_vec # Unpack the coordinates of the input position vector into x and y components
        return x >= self.x_min and x < self.x_max and y >= self.y_min and y <= self.y_max and z >= self.z_min and z <= self.z_max
        
    def insert(self, pos_vec:np.ndarray, mass_val:float):
        """Method for adding a new particle to the BH tree, recursively creates new octants"""
        if self.particle_count == 0 and self.contains(pos_vec): # Check if the particle is cotained in this node
            self.center_of_mass = pos_vec # Record the position of the particle as the center of mass
            self.total_mass = mass_val # Record the mass of the particle
            self.particle_count += 1 # Incriment up the particle count for this node (and all of its children) 
            
        elif self.particle_count == 1: # If the particle count is equal to 1, then we have to convert this node from an
            # external or leaf node into an internal node (i.e. one with child nodes). First, create a series of child
            # nodes for a new octant sub-division and pass down the existing particle into the appropriate octant
            # Create new child nodes in the tree using the octant_node class and create smaller boundary divisions
            # Compute the midpoints of the current cube region
            x_mid, y_mid, z_mid = (self.x_min+self.x_max)/2, (self.y_min+self.y_max)/2, (self.z_min+self.z_max)/2
            # Front face of the cube cube, the ones with the least depth along the z-axis
            self.child_NW1 = octant_node(self.x_min,x_mid,y_mid,self.y_max,self.z_min,z_mid) 
            self.child_NE1 = octant_node(x_mid,self.x_max,y_mid,self.y_max,self.z_min,z_mid)
            self.child_SE1 = octant_node(x_mid,self.x_max,self.y_min,y_mid,self.z_min,z_mid)
            self.child_SW1 = octant_node(self.x_min,x_mid,self.y_min,y_mid,self.z_min,z_mid)
            
            # Back half of the cube cube, the ones with the most depth along the z-axis
            self.child_NW2 = octant_node(self.x_min,x_mid,y_mid,self.y_max,z_mid,self.z_max) 
            self.child_NE2 = octant_node(x_mid,self.x_max,y_mid,self.y_max,z_mid,self.z_max)
            self.child_SE2 = octant_node(x_mid,self.x_max,self.y_min,y_mid,z_mid,self.z_max)
            self.child_SW2 = octant_node(self.x_min,x_mid,self.y_min,y_mid,z_mid,self.z_max)
            
            # Next, find which new sub-node octant the current particle stored in this node can be passed to
            self._pass_to_child(self.center_of_mass, self.total_mass)
            # Next update the center-of-mass and total mass at this node
            self.center_of_mass = (self.center_of_mass*self.total_mass + pos_vec*mass_val) / (self.total_mass + mass_val)
            self.total_mass += mass_val # Update the total mass
            self.particle_count +=1 # Update the particle count
            self._pass_to_child(pos_vec, mass_val) # Now pass down the new particle down to a child sub-octant
        
        else: # Otherwise, if this node already has particles in it, then update the center-of-mass and total mass at this node
            self.center_of_mass = (self.center_of_mass*self.total_mass + pos_vec*mass_val) / (self.total_mass + mass_val)
            self.total_mass += mass_val # Incriment up the total mass of this node based on the new point added 
            self.particle_count += 1 # Record the number of points within this octant, add 1 for the new particle
            # At this point with 2 or more particles, there will be child sub-quadrants, pass the new particle into the next layer
            self._pass_to_child(pos_vec, mass_val)

    def _pass_to_child(self, pos_vec:np.ndarray, mass_val:float):
        """Internal helper method for passing a new particle down into a node's child sub-octants"""
        if self.child_NW1.contains(pos_vec): # Check for each octant where this new particle belongs
            self.child_NW1.insert(pos_vec,mass_val)
        elif self.child_NE1.contains(pos_vec):
            self.child_NE1.insert(pos_vec,mass_val)
        elif self.child_SE1.contains(pos_vec):
            self.child_SE1.insert(pos_vec,mass_val)
        elif self.child_SW1.contains(pos_vec):
            self.child_SW1.insert(pos_vec,mass_val)
        
        elif self.child_NW2.contains(pos_vec):
            self.child_NW2.insert(pos_vec,mass_val)
        elif self.child_NE2.contains(pos_vec):
            self.child_NE2.insert(pos_vec,mass_val)
        elif self.child_SE2.contains(pos_vec):
            self.child_SE2.insert(pos_vec,mass_val)
        elif self.child_SW2.contains(pos_vec):
            self.child_SW2.insert(pos_vec,mass_val)
    
        else:
            raise AttributeError("No suitable sub-octant found for new particle:",pos_vec)
    
    def plot_boundaries(self,ax=None,col:str='black',linewidth=1):
        """A recursive method for plotting the octant sub-divisions made by the algorithm"""
        if ax is None:
            fig,ax = plt.subplots(1,1)
            ax = fig.add_subplot(projection='3d') # Create 3d projection sub-axis
        
        if self.particle_count > 1: # If there are other particles, then recursively call plot_boundaries on child nodes
            self.child_NW1.plot_boundaries(ax,col,linewidth)
            self.child_NE1.plot_boundaries(ax,col,linewidth)
            self.child_SE1.plot_boundaries(ax,col,linewidth)
            self.child_SW1.plot_boundaries(ax,col,linewidth)
            self.child_NW2.plot_boundaries(ax,col,linewidth)
            self.child_NE2.plot_boundaries(ax,col,linewidth)
            self.child_SE2.plot_boundaries(ax,col,linewidth)
            self.child_SW2.plot_boundaries(ax,col,linewidth)
        
        else: # If no child nodes, then plot the boundaries of this node
            # Front face of cube
            ax.plot([self.x_min,self.x_max],[self.y_max,self.y_max],[self.z_min,self.z_min],color=col,linewidth=linewidth) # Add the upper border
            ax.plot([self.x_min,self.x_max],[self.y_min,self.y_min],[self.z_min,self.z_min],color=col,linewidth=linewidth) # Add the bottom border
            ax.plot([self.x_min,self.x_min],[self.y_min,self.y_max],[self.z_min,self.z_min],color=col,linewidth=linewidth) # Add the left border
            ax.plot([self.x_max,self.x_max],[self.y_min,self.y_max],[self.z_min,self.z_min],color=col,linewidth=linewidth) # Add the right border
            
            # Back face of cube
            ax.plot([self.x_min,self.x_max],[self.y_max,self.y_max],[self.z_max,self.z_max],color=col,linewidth=linewidth) # Add the upper border
            ax.plot([self.x_min,self.x_max],[self.y_min,self.y_min],[self.z_max,self.z_max],color=col,linewidth=linewidth) # Add the bottom border
            ax.plot([self.x_min,self.x_min],[self.y_min,self.y_max],[self.z_max,self.z_max],color=col,linewidth=linewidth) # Add the left border
            ax.plot([self.x_max,self.x_max],[self.y_min,self.y_max],[self.z_max,self.z_max],color=col,linewidth=linewidth) # Add the right border
            
            # Side edges of cube
            ax.plot([self.x_min,self.x_min],[self.y_max,self.y_max],[self.z_min,self.z_max],color=col,linewidth=linewidth) # Top left
            ax.plot([self.x_max,self.x_max],[self.y_max,self.y_max],[self.z_min,self.z_max],color=col,linewidth=linewidth) # Top right
            ax.plot([self.x_min,self.x_min],[self.y_min,self.y_min],[self.z_min,self.z_max],color=col,linewidth=linewidth) # Bottom left
            ax.plot([self.x_max,self.x_max],[self.y_min,self.y_min],[self.z_min,self.z_max],color=col,linewidth=linewidth) # Bottom right
            

    def check_structure(self):
        """Helper method for checking that the tree has been properly constructed"""
        assert self.total_mass == self._get_descendent_mass() # Check that the total mass matches the sum of the descendants' mass
        assert self.particle_count == self._get_descendant_particles() # Check that the total particle count matches
        # the sum of particles across all of the descendants
        print("All checks passed!")
        return True
    
    def _get_descendent_mass(self):
        """Helper function for check_structure to gather descendant node masses"""
        if self.particle_count<=1: # If 1 or 0 particles in this octant, then return the total_mass of the node
            return self.total_mass
        else: # Otherwise return the sum overall the descendant nodes
            return self.child_NE1._get_descendent_mass() + self.child_NW1._get_descendent_mass() + \
                self.child_SE1._get_descendent_mass() + self.child_SW1._get_descendent_mass() + \
                self.child_NE2._get_descendent_mass() + self.child_NW2._get_descendent_mass() + \
                self.child_SE2._get_descendent_mass() + self.child_SW2._get_descendent_mass()
    
    def _get_descendant_particles(self): 
        """Helper function for check_structure to gather descendant node particles"""
        if self.particle_count<=1: # If 1 or 0 particles in this octant, then return the particle_count of the node
            return self.particle_count
        else: # Otherwise return the sum overall the descendant nodes
            return self.child_NE1._get_descendant_particles() + self.child_NW1._get_descendant_particles() + \
                self.child_SE1._get_descendant_particles() + self.child_SW1._get_descendant_particles() + \
                self.child_NE2._get_descendant_particles() + self.child_NW2._get_descendant_particles() + \
                self.child_SE2._get_descendant_particles() + self.child_SW2._get_descendant_particles()
    
    def get_traversal(self, pos_vec:np.ndarray, theta:float=0.5, assert_check=False):
        """Returns a pos np.array and a mass np.array containing the positions (x,y) and a masses of a set of particles
           to be used in the net acceleration calculation based on the Barnes-Hut proximity rule that treates distant
           masses and 1 combined center of mass. Theta controls the extent to which we summarize particles as 1 mass.
           If s (the width of the node's region) divided by d (the distance to the center of mass of that node) is
           less than theta, then the internal node is sufficiently far away to treat as 1 center-of-mass. If theta==0,
           then we will never perform any summarizing, this method will return the positions of all particles. If theta
           is very large, then this method will return just 1 particle i.e. all other summarized as 1 node. 
        """       
        if self.particle_count == 1: # If there is only 1 particle in the node octant, return it
            return [self.center_of_mass], [self.total_mass]
        
        elif self.particle_count > 1: # If there is more than just 1 particle, determine if we should treat them as 1 mass
            s = ((self.x_max - self.x_min) + (self.y_max - self.y_min) + (self.z_max - self.z_min))/3 # Compute a width for this octant as the average
            # width along the x and y dimensions since sub-quadrants may not necessarily be square
            # Compute the distance from the input pos_vector to the node's center-of-mass
            d = self.euclidian_dist(pos_vec, self.center_of_mass)
            if s / d < theta: # If the octant is sufficiently small relative to the distance away, treat it as 1 combined mass
                return [self.center_of_mass], [self.total_mass]
            
            else: # Otherwise, recursively traverse the tree to build a set of position vectors and mass values
                pos_agg = [];mass_agg=[] # Create blank lists to hold the outputs from the recursive calls
                
                if self.child_NW1.particle_count>0: # Check if the child node is non-empty
                    pos_list, mass_list = self.child_NW1.get_traversal(pos_vec, theta) # Get relevant points from sub-octant
                    pos_agg+=pos_list;mass_agg+=mass_list # Append output from recursive traversal to aggregation lists
                
                if self.child_NE1.particle_count>0: # Check if the child node is non-empty
                    pos_list, mass_list = self.child_NE1.get_traversal(pos_vec, theta) # Get relevant points from sub-octant
                    pos_agg+=pos_list;mass_agg+=mass_list # Append output from recursive traversal to aggregation lists
                
                if self.child_SE1.particle_count>0: # Check if the child node is non-empty
                    pos_list, mass_list = self.child_SE1.get_traversal(pos_vec, theta) # Get relevant points from sub-octant
                    pos_agg+=pos_list;mass_agg+=mass_list # Append output from recursive traversal to aggregation lists
                
                if self.child_SW1.particle_count>0: # Check if the child node is non-empty
                    pos_list, mass_list = self.child_SW1.get_traversal(pos_vec, theta) # Get relevant points from sub-octant
                    pos_agg+=pos_list;mass_agg+=mass_list # Append output from recursive traversal to aggregation lists

                if self.child_NW2.particle_count>0: # Check if the child node is non-empty
                    pos_list, mass_list = self.child_NW2.get_traversal(pos_vec, theta) # Get relevant points from sub-octant
                    pos_agg+=pos_list;mass_agg+=mass_list # Append output from recursive traversal to aggregation lists
                
                if self.child_NE2.particle_count>0: # Check if the child node is non-empty
                    pos_list, mass_list = self.child_NE2.get_traversal(pos_vec, theta) # Get relevant points from sub-octant
                    pos_agg+=pos_list;mass_agg+=mass_list # Append output from recursive traversal to aggregation lists
                
                if self.child_SE2.particle_count>0: # Check if the child node is non-empty
                    pos_list, mass_list = self.child_SE2.get_traversal(pos_vec, theta) # Get relevant points from sub-octant
                    pos_agg+=pos_list;mass_agg+=mass_list # Append output from recursive traversal to aggregation lists
                
                if self.child_SW2.particle_count>0: # Check if the child node is non-empty
                    pos_list, mass_list = self.child_SW2.get_traversal(pos_vec, theta) # Get relevant points from sub-octant
                    pos_agg+=pos_list;mass_agg+=mass_list # Append output from recursive traversal to aggregation lists

                if assert_check:
                    assert len(pos_agg) == len(mass_agg)
                return pos_agg, mass_agg
        
    def euclidian_dist(self, pos_vec1, pos_vec2):
        """Computes the Euclidian distance between 2 position vectors"""
        return np.linalg.norm(pos_vec1 - pos_vec2)
    
    def __str__(self):
        """A string representation method used to print out the tree structure when called through print"""
        if self.particle_count==0:
            return "-NA"
        elif self.particle_count==1:
            return f'BH-Tree({self.particle_count}, {self.total_mass})'
        else:
            return f'BH-Tree({self.particle_count}, {self.total_mass})' + \
                   '\n|\n|-(NW1)->' + '\n|      '.join(str(self.child_NW1).split('\n')) + \
                   '\n|\n|-(NE1)->' + '\n|      '.join(str(self.child_NE1).split('\n')) + \
                   '\n|\n|-(SE1)->' + '\n|      '.join(str(self.child_SE1).split('\n')) + \
                   '\n|\n|-(SW1)->' + '\n|      '.join(str(self.child_SW1).split('\n')) + \
                   '\n|\n|-(NW2)->' + '\n|      '.join(str(self.child_NW2).split('\n')) + \
                   '\n|\n|-(NE2)->' + '\n|      '.join(str(self.child_NE2).split('\n')) + \
                   '\n|\n|-(SE2)->' + '\n|      '.join(str(self.child_SE2).split('\n')) + \
                   '\n|\n|-(SW2)->' + '\n|      '.join(str(self.child_SW2).split('\n'))



###################################
### Testing the BH 3d algorithm ###
###################################

if __name__ == "__main__":
    # Generate a sample data set
    n = 15 # The number of particles to simulate
    pos = np.random.random(n*3).reshape(n,3) # A n x 3 vector of (x,y,z) position coordinates for n particles
    mass = np.ones(len(pos)).reshape(-1,1) # Give each particle a mass of 1
    
    BH_tree = octant_node(0,1,0,1,0,1) # Initialize a BH tree from x:[0,1], y:[0:1], z:[0:1], make sure to initialize so
    # that all of the particles are contained within the initial node space, otherwise we will have an error
    # try to round up to the next nearest integer for that to happen
    
    for particle_pos,particle_mass in zip(pos, mass.reshape(-1)): # Iterate over all the particles
        BH_tree.insert(particle_pos,particle_mass) # Add each one to the BH tree

    # Check that the tree contains all the mass and particles we expect it to
    assert BH_tree.check_structure();assert BH_tree.particle_count == len(pos);assert BH_tree.total_mass == mass.sum()
    print(BH_tree) # Print the string representation of the Barnes Hut Tree
    
    # Plot the data points along with the boundaries in 2D
    fig,axes = plt.subplots(1,1,figsize=(6,6))
    axes.axis('off') # Remove x and y axis labels on outer plot area and outer box
    axes = fig.add_subplot(projection='3d') # Create 3d projection sub-axis
    axes.scatter(pos[:,0],pos[:,1],pos[:,2],s=20)    
    axes.set_xlabel("X");axes.set_ylabel("Y");axes.set_zlabel("Z")
    axes.set_title("Barnes-Hut Tree Sub-Set Bounaries")
    #axes.plot([1,0.5],[1,0.5],[1,0.5],color='darkorange',linewidth=1)
    BH_tree.plot_boundaries(axes,linewidth=0.25) # Add Barnes-Hut algorithm bounary lines
    
    # Testing the tree-traversal
    test_point = np.array([0.5,0.5,0.5]);theta=1 # Generate a test point and a theta value
    pos_agg, mass_agg = BH_tree.get_traversal(test_point, theta)
    
    pos_agg = np.stack(pos_agg);mass_agg = np.array(mass_agg) # Convert to np.arrays
    print("pos_agg.shape",pos_agg.shape);print("mass_agg.shape",mass_agg.shape)
    
    # Plot a visualization of the original data set and the Barnes-Hut reduced data set
    fig,axes = plt.subplots(1,1,figsize=(6,6))
    axes.axis('off') # Remove x and y axis labels on outer plot area and outer box
    axes = fig.add_subplot(projection='3d') # Create 3d projection sub-axis
    axes.scatter(pos[:,0],pos[:,1],pos[:,2],label="Original");axes.set_xlabel("X");axes.set_ylabel("Y");axes.set_zlabel("Z")
    axes.scatter(pos_agg[:,0],pos_agg[:,1],pos_agg[:,2],label="BH Reduction",marker="^")
    axes.scatter(test_point[0],test_point[1],test_point[2],marker=",",label="Eval Point");axes.legend()
    axes.set_title("Barnes-Hut Point Reduction - Theta="+str(theta))
    #BH_tree.plot_boundaries(axes) # Add Barnes-Hut algorithm bounary lines
    plt.show()






