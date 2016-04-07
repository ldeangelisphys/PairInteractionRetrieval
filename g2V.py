import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import time
from mpl_toolkits.mplot3d import Axes3D

class par:
    def __init__(self):
        self.r = 0
        self.v = 0
        self.bin = 0


def pair_correlation_function_2D(kind, x, y, S, rMax, dr): 
    """
    Compute the two-dimensional pair correlation function, also known 
    as the radial distribution function, for a set of circular particles 
    contained in a square region of a plane.  This simple function finds 
    reference particles such that a circle of radius rMax drawn around the 
    particle will fit entirely within the square, eliminating the need to 
    compensate for edge effects.  If no such particles exist, an error is 
    returned. Try a smaller rMax...or write some code to handle edge effects! ;) 

    Arguments: 
        x               an array of x positions of centers of particles 
        y               an array of y positions of centers of particles 
        S               length of each side of the square region of the plane 
        rMax            outer diameter of largest annulus 
        dr              increment for increasing radius of annulus 
	
    Returns a tuple: (g, radii, interior_indices) 
        g(r)            a numpy array containing the correlation function g(r) 
        radii           a numpy array containing the radii of the 
                        annuli used to compute g(r) 
                        reference_indices   indices of reference particles 
    """ 
    # Number of particles in ring/area of ring/number of reference particles/number density 
    # area of ring = pi*(r_outer**2 - r_inner**2) 
    
    
    # Find particles which are close enough to the box center that a circle of radius 
    # rMax will not cross any edge of the box 
    bools1 = x > rMax 
    bools2 = x < (S - rMax) 
    bools3 = y > rMax 
    bools4 = y < (S - rMax) 
    interior_indices, = np.where(bools1 * bools2 * bools3 * bools4) 
    num_interior_particles = len(interior_indices) 
    
    
    if num_interior_particles < 1: 
        raise  RuntimeError ("No particles found for which a circle of radius rMax will lie entirely within a square of side length S.  Decrease rMax or increase the size of the square.") 


    edges = np.arange(0., rMax + 1.1 * dr, dr) 
    num_increments = len(edges) - 1 
    g = np.zeros([num_interior_particles, num_increments]) 
    radii = np.zeros(num_increments) 
    numberDensity = len(x) / np.double(S**2)
    theta = np.pi/8
    slice_angle = 2*theta
    ang_frac = np.pi/slice_angle
    
    # Compute pairwise correlation for each interior particle 
    for p in range(num_interior_particles): 
        index = interior_indices[p]
        
        d = np.sqrt((x[index] - x)**2 + (y[index] - y)**2)
        d[index] = 2 * rMax

        # Here I calculate the angle of the vector between the two particles
        # I'm here considering simmetry along x and y axis
        # By only using the absolute values of the distances
        if((kind == 'unsigned_y')|(kind == 'signed_y')):
            angles = np.arctan2(np.abs(x[index] - x),np.abs(y[index] - y))
        else:
            angles = np.arctan2(np.abs(y[index] - y),np.abs(x[index] - x))            
        # And I select only the ones that are within a certain angle theta
        angular_selection = np.where(angles < theta)
        d_angular = d[angular_selection]        
        
        if(kind == 'unsigned'):
            (result, bins) = np.histogram(d, bins=edges, density=False) 
            g[p, :] = result/numberDensity
        elif((kind == 'unsigned_x')|(kind == 'unsigned_y')):
            (result,bins) = np.histogram(d_angular, bins = edges, density = False)
            g[p, :] = (result/numberDensity)*ang_frac
            
        else:
            sign_product = sign[index]*sign
            
            pos_indices = np.where(sign_product == 1)
            neg_indices = np.where(sign_product == -1)
            
            d_pos = d[pos_indices]
            d_neg = d[neg_indices]

            (pos_result, pos_bins) = np.histogram(d_pos, bins=edges, density=False)
            (neg_result, neg_bins) = np.histogram(d_neg, bins=edges, density=False)
            

            if(kind == 'signed'):
                g[p, :] = (pos_result-neg_result)/numberDensity
            elif(kind == 'same_sign'):
                numberDensity = len(d_pos) / np.double(S**2)
                g[p, :] = pos_result/numberDensity
            elif(kind == 'opp_sign'):
                numberDensity = len(d_neg) / np.double(S**2)
                g[p, :] = neg_result/numberDensity
            elif((kind == 'signed_x')|(kind == 'signed_y')):
                # For the angular case
                pos_angular_indices = np.where((sign_product == 1)&(angles < theta))
                neg_angular_indices = np.where((sign_product == -1)&(angles < theta))

                d_angular_pos = d[pos_angular_indices]
                d_angular_neg = d[neg_angular_indices]

                (pos_angular_result, pos_bins) = np.histogram(d_angular_pos, bins=edges, density=False)
                (neg_angular_result, neg_bins) = np.histogram(d_angular_neg, bins=edges, density=False)                        
            
                g[p, :] = ((pos_angular_result-neg_angular_result)/numberDensity)*ang_frac
            
            else:
                sys.exit('Error! ' + kind + ' g(r): definition not found.')

    # Average g(r) for all interior particles and compute radii 
    g_average = np.zeros(num_increments) 
    for i in range(num_increments): 
        radii[i] = (edges[i] + edges[i+1]) / 2. 
        rOuter = edges[i + 1] 
        rInner = edges[i] 
        g_average[i] = np.mean(g[:, i]) / (np.pi * (rOuter**2 - rInner**2)) 
	
	
    return (g_average, radii)
    
    
def replicate_3D(particles,L_box):
    N,dim = np.shape(particles)
    shifts = [[0,0,1],[0,0,2],[0,1,0],[0,1,1],[0,1,2],[0,2,0],[0,2,1],[0,2,2],[1,0,0],[1,0,1],[1,0,2],[1,1,0],[1,1,1],[1,1,2],[1,2,0],[1,2,1],[1,2,2],[2,0,0],[2,0,1],[2,0,2],[2,1,0],[2,1,1],[2,1,2],[2,2,0],[2,2,1],[2,2,2]]
    more_particles = particles
    for delta in shifts:
        more_particles = np.append(more_particles,particles + L_box*np.array(delta),axis = 0)
    return more_particles


def pair_correlation_function_3D(x, y, z, S, rMax, dr):
    """Compute the three-dimensional pair correlation function for a set of
    spherical particles contained in a cube with side length S.  This simple
    function finds reference particles such that a sphere of radius rMax drawn
    around the particle will fit entirely within the cube, eliminating the need
    to compensate for edge effects.  If no such particles exist, an error is
    returned.  Try a smaller rMax...or write some code to handle edge effects! ;)
    Arguments:
        x               an array of x positions of centers of particles
        y               an array of y positions of centers of particles
        z               an array of z positions of centers of particles
        S               length of each side of the cube in space
        rMax            outer diameter of largest spherical shell
        dr              increment for increasing radius of spherical shell
    Returns a tuple: (g, radii, interior_indices)
        g(r)            a numpy array containing the correlation function g(r)
        radii           a numpy array containing the radii of the
                        spherical shells used to compute g(r)
        reference_indices   indices of reference particles
    """
    from numpy import zeros, sqrt, where, pi, mean, arange, histogram

    # Find particles which are close enough to the cube center that a sphere of radius
    # rMax will not cross any face of the cube
    bools1 = x > rMax
    bools2 = x < (S - rMax)
    bools3 = y > rMax
    bools4 = y < (S - rMax)
    bools5 = z > rMax
    bools6 = z < (S - rMax)

    interior_indices, = where(bools1 * bools2 * bools3 * bools4 * bools5 * bools6)
    num_interior_particles = len(interior_indices)

    if num_interior_particles < 1:
        raise  RuntimeError ("No particles found for which a sphere of radius rMax\
                will lie entirely within a cube of side length S.  Decrease rMax\
                or increase the size of the cube.")

    edges = arange(0., rMax + 1.1 * dr, dr)
    num_increments = len(edges) - 1
    g = zeros([num_interior_particles, num_increments])
    radii = zeros(num_increments)
    numberDensity = len(x) / S**3

    # Compute pairwise correlation for each interior particle
    for p in range(num_interior_particles):
        index = interior_indices[p]
        d = sqrt((x[index] - x)**2 + (y[index] - y)**2 + (z[index] - z)**2)
        d[index] = 2 * rMax

        (result, bins) = histogram(d, bins=edges, normed=False)
        g[p,:] = result / numberDensity

    # Average g(r) for all interior particles and compute radii
    g_average = zeros(num_increments)
    for i in range(num_increments):
        radii[i] = (edges[i] + edges[i+1]) / 2.
        rOuter = edges[i + 1]
        rInner = edges[i]
        g_average[i] = mean(g[:, i]) / (4.0 / 3.0 * pi * (rOuter**3 - rInner**3))

    return (g_average, radii, interior_indices)
    # Number of particles in shell/total number of particles/volume of shell/number density
    # shell volume = 4/3*pi(r_outer**3-r_inner**3)
####    
    
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
        particles = np.random.rand(N_particles,dim)*L_box
    
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
    

    
def MC_sim(particles,L_box,N_iterations,v,R_cut):
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
        old_histo,bins  = np.histogram(old_distances, bins = v.bin)
        new_histo,bins  = np.histogram(new_distances, bins = v.bin)
        dE = np.sum((new_histo-old_histo)*v.v)
        #dE = np.sum(potential(new_distances)) - np.sum(potential(old_distances))
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
            
def test_montecarlo():
    """ Test the MC simulation using the example in hte paper byLyubartsev and Laaksonen"""
    
    v = par()    
    
    v.r,v.v = get_g('/home/deangelis/DATA/ReverseMC/vtest.dat')
    v.bin = np.append(0,np.append(0.5*(v.r[1:]+v.r[:-1]),float('Inf')))  
    #v_f = interp1d(r,v)

    L_box = 20
    N_particles = 50
    dim = 3
    # Initialize the system at a random distribution
    starting = initialize_system(N_particles,L_box,dim,'random')

    particles,E = MC_sim(starting,L_box,100,v,v.r[-1])
    
    
    #Replicate the system that I considered periodic
    more_particles = replicate_3D(particles,20)
    #plot it    
    [xp,yp,zp] = np.transpose(more_particles)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xp, yp, zp)
    # Calculate pair correlation function
    gm,rm,im = pair_correlation_function_3D(xp,yp,zp,L_box*3.,9.75,0.5)
    #And import the paper one
    rt,gt = get_g('/home/deangelis/DATA/ReverseMC/gtest.dat')
    #plot it together with the one given by the paper
    plt.figure()
    plt.plot(rm,gm)
    plt.plot(rt,gt)
    
    return particles,E
    
    
            
if __name__ == '__main__':

    start = time.time()
    particles,E = test_montecarlo()    

    
    
    end = time.time()
    duration = end - start
    print 'Done in %.1f s' % duration
