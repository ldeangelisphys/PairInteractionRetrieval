import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import time

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

def lj(r):
    """Lennard-Jones potential"""
    sigma = 5.
    eps = 2.    
    
    return 4*eps*((sigma/r)**12-(sigma/r)**6)   
    
    
def MC_sim(N_particles,L_box,dim,N_iterations,potential,R_cut):
    """Performs a Monte Carlo simulation"""

    # Initialize the system
    particles = initialize_system(N_particles,L_box,dim,'random')
    E = np.zeros(N_iterations+1) #TODO    
    
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
        old_distances = np.sqrt(np.sum((other_particles - old_particle)**2,axis = 1))
        new_distances = np.sqrt(np.sum((other_particles - new_particle)**2,axis = 1))
        # Particles at a distance > R_cut don't contribute to the energy
        old_distances = old_distances[np.where(old_distances < R_cut)]
        new_distances = new_distances[np.where(new_distances < R_cut)]
        dE = np.sum(potential(new_distances)) - np.sum(potential(old_distances))
        #Accept or decline the movement
        acc_prob = np.min([1,np.exp(-dE)])
        if random_extraction[n] < acc_prob:
            # perform the movement
            particles[chosen_one] += dr 
            # update observables
            E[n+1] = E[n] + dE
            # And count the MC move
            MC_move += 1
        else:
            E[n+1] = E[n]
    

    print MC_move

    return particles,E
            
            
            
if __name__ == '__main__':


    r,v = get_g('/home/deangelis/DATA/ReverseMC/vtest.dat')
    v_f = interp1d(r,v)

    start = time.time()
    particles,E = MC_sim(50,20,2,10000000,v_f,r[-1])
    end = time.time()
    duration = end - start
    print duration
