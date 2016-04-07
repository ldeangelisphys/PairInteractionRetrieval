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
  
def calc_distances(elements,ref,L_box,R_cut):
    
    starting_shape = np.shape(elements)
    # Calculate the dx and dy in the next unit cell (+- L_box)
    delta2 = np.reshape((elements - ref)**2,-1)
    delta2_p = np.reshape((elements - ref + L_box)**2,-1)
    delta2_m = np.reshape((elements - ref - L_box)**2,-1)
    # And take the distance that is minimum in each direction
    delta_min = np.min([delta2,delta2_p,delta2_m], axis = 0)
    # Reshape the array in the old form, so that we can work on it
    delta_min = np.reshape(delta_min,starting_shape)
    # and calculate the modulus of the distance between particles
    distances = np.sqrt(np.sum(delta_min,axis = 1))
    # And finally exclude particles at a distance bigger than R_cut
    distances = distances[np.where(distances < R_cut)]

    return distances    
    
    
    
def MC_sim(particles,L_box,N_iterations,potential,R_cut):
    """Performs a Monte Carlo simulation"""

    (N_particles,dim) = np.shape(particles)
    E = np.zeros(N_iterations+1) #TODO initial energy       
    
    
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
        new_particle = (old_particle + dr)%L_box
        # Apply periodic Boundary conditions and exclude particles outside R_cut
        # Particles at a distance > R_cut don't contribute to the energy
        old_distances = calc_distances(other_particles,old_particle,L_box,R_cut)   
        new_distances = calc_distances(other_particles,new_particle,L_box,R_cut)
        dE = np.sum(potential(new_distances)) - np.sum(potential(old_distances))
        #Accept or decline the movement
        acc_prob = np.min([1,np.exp(-dE)])
        if random_extraction[n] < acc_prob:
            # perform the movement
            particles[chosen_one] = new_particle
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

    L_box = 20
    N_particles = 50
    dim = 2

    # Initialize the system at a random distribution
    starting = initialize_system(N_particles,L_box,dim,'random')
    [xi,yi] = np.transpose(starting)
    plt.scatter(xi,yi)

    start = time.time()
    particles,E = MC_sim(starting,L_box,1000,v_f,r[-1])
    end = time.time()
    duration = end - start
    print duration

    plt.figure()
    [x,y] = np.transpose(particles)
    plt.scatter(x,y, c = 'g')