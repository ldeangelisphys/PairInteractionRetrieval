import numpy as np
import matplotlib.pyplot as plt

def get_g(file_name):
    """Gets the radial distribution function from file_name"""
    
    r,g = np.loadtxt(file_name, dtype = 'float', unpack = 'true')
    
    return r,g
    
def init_V(g):
    """Initialize the potential with the effective potential function approximation"""
    return -np.log(g)

def initialize_system(N_particles,L_box,dim,how):
    """Initialize a list of positions of N particles in a box of size L"""
    
    if how == 'random':
        particles = np.random.rand(N_particles,2)*L_box
    
    return particles
    
def MC_sim():
    """Performs a Monte Carlo simulation"""

    # Initialize the system
    particles = initialize_system(N_particles,L_box,dim,'random')

    for n in range(N_iterations):
        
        #Propose a movement
        i = random_particle
        #Calculate the difference in energy
        other_particles = particles
        other_particles.pop(i)
        old_particle = particles[i]
        new_particle = particles[i] + dr
        dE = np.sum(potential(np.abs(other_particles - new_particle)) - potential(np.abs(other_particles - old_particle)))
        #Accept or decline the movement
        acc_prob = np.min(1,np.exp(-dE))
        if random_list[n] < acc_prob:
            # perform the movement
            particles[i] = particles[i] + dr 
            # update observables
            E += dE
            # And count the MC move
            MC_move += 1
    
    
    return particles
            