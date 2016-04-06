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
    """Initialize an array of positions of N particles in a box of size L"""
    
    if how == 'random':
        particles = np.random.rand(N_particles,2)*L_box
    
    return particles
    
def MC_sim(N_particles,L_box,dim,N_iterations):
    """Performs a Monte Carlo simulation"""

    # Initialize the system
    particles = initialize_system(N_particles,L_box,dim,'random')
    E = 0 #TODO    
    
    # Initialize N_iterations random extractions
    random_extraction = np.random.rand(N_iterations)
    
    MC_move = 0

    for n in range(N_iterations):
        
        #Propose a movement
        chosen_one = np.random.randint(N_particles)
        dr = np.random.rand(dim)
        #Calculate the difference in energy
        other_particles = np.delete(particles,chosen_one,0)
        old_particle = particles[chosen_one]
        new_particle = old_particle + dr
        dE = np.sum(potential(np.abs(other_particles - new_particle)) - potential(np.abs(other_particles - old_particle)))
        #Accept or decline the movement
        acc_prob = np.min(1,np.exp(-dE))
        if random_extraction[n] < acc_prob:
            # perform the movement
            particles[chosen_one] += dr 
            # update observables
            E += dE
            # And count the MC move
            MC_move += 1
    
    
    return particles
            