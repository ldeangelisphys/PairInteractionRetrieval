import numpy as np
import matplotlib.pyplot as plt
#from scipy.interpolate import interp1d
import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import subprocess
import os

class par:
    def __init__(self):
        self.r = 0
        self.v = 0
        self.bin = 0

def check_folders_existence(f_path):    
    
    folders_list = ['final_g','final_configuration','final_g/all_g_%dmcs' % N_mcs]

    for folder in folders_list:        
        if not os.path.exists(f_path + folder):
            os.makedirs(f_path + folder)
            
    return

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
    elif how == 'array' or 'array_w_noise':
        n = np.power(N_particles,1.0/dim)
        n = int(n) + 1
        n_generated = n**dim
        X,Y,Z = np.mgrid[0:n,0:n,0:n]
        more_particles = np.array([X.flatten(),Y.flatten(),Z.flatten()]).T
        n_excess = n_generated - N_particles
        # Remove the particles in excess (randomly)
        to_remove = np.random.permutation(n_generated)[:n_excess]
        particles = np.delete(more_particles, to_remove, axis = 0)
        # normalize
        particles = particles * L_box / n
        
        if how == 'array_w_noise':
            noise = np.random.rand(N_particles,dim) - 0.5
            particles = particles + noise
            
                
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
    
def MC_step(particles,chosen_one,dr,R_cut,v):
    """Performs a single Monte Carlo step"""
    
    #%%Calculate the difference in energy
    other_particles = np.delete(particles,chosen_one,0)
    old_particle = particles[chosen_one]
    new_particle = (old_particle + dr)%L_box
    #%% Apply periodic Boundary conditions and exclude particles outside R_cut
    # Particles at a distance > R_cut don't contribute to the energy
    old_distances = calc_distances(other_particles,old_particle,L_box,R_cut)   
    new_distances = calc_distances(other_particles,new_particle,L_box,R_cut)
    old_histo,bins  = np.histogram(old_distances, bins = v.bin)
    new_histo,bins  = np.histogram(new_distances, bins = v.bin)
    dE = np.sum((new_histo-old_histo)*v.v)
    #%%dE = np.sum(potential(new_distances)) - np.sum(potential(old_distances))
    #Accept or decline the movement
    acc_prob = np.min([1,np.exp(-dE)])
    if np.random.rand() < acc_prob:
        # perform the movement
        particles[chosen_one] = new_particle
        move = 1
    else:
        # Decline
        move = 0
        dE = 0

    return dE,move
            
def run_montecarlo(v, n_run, dr_coeff = 0.58):
    """ Test the MC simulation using the example in hte paper byLyubartsev and Laaksonen"""
    
    start = time.time()

    R_cut = v.r[-1]  # TODO
    #v_f = interp1d(r,v)

    g_list = []    
    
    # Initialize the system at a random distribution
    particles = initialize_system(N_particles,L_box,dim,'array_w_noise')
    E = np.zeros(N_mcs+2) #TODO initial energy
    MC_move = 0

    for n in range(N_mcs+1):
        
        chosen_one = np.random.randint(N_particles)
        dr = dr_coeff*np.random.rand(dim)
        ####
        dE,this_move = MC_step(particles,chosen_one,dr,R_cut,v)
        ####
        E[n+1] = E[n] + dE
        MC_move += this_move
        
        ## After convergence calc and save g every N_corr steps
        if((n > N_conv)&(n%N_corr==0)):
            gmeas = calc_and_plot_g_r(particles,n)
            g_list.append(gmeas)
            
    elapsed = time.time() - start
    print('Done in %d s' % elapsed)
    
    print('%d %% of the Monte Carlo steps were performed (%d out of %d)' % (100.0*MC_move/N_mcs, MC_move,N_mcs))

    plot_conf(particles,N_mcs,i)

        
    return particles,E,g_list
    
def calc_and_plot_g_r(particles,N_iter):
    
    #Replicate the system that I considered periodic
    more_particles = replicate_3D(particles,20)
    #plot it    
    [xp,yp,zp] = np.transpose(more_particles)
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d')
#    ax.scatter(xp, yp, zp)
    # Calculate pair correlation function
    gmeas = par()
    gmeas.v,gmeas.r,_ = pair_correlation_function_3D(xp,yp,zp,L_box*3.,9.75,0.5)
    #And import the paper one
    gtheory = par()
    gtheory.r,gtheory.v = get_g('D:/Google Drive/Potential Retrieval/gtest.txt')       #plot it together with the one given by the paper
    plt.figure(figsize = (7,4))
    plt.plot(gmeas.r,gmeas.v)
    plt.plot(gtheory.r,gtheory.v)
    plt.xlabel('r')
    plt.ylabel('g(r)')
    plt.figtext(0.99, 0.99, git_v, fontsize = 8, ha = 'right', va = 'top')
    plt.savefig('D:/Google Drive/Potential Retrieval/final_g/all_g_%dmcs/n%d.png' % (N_mcs,N_iter), dpi = 300)
    plt.close('all')
    
    return gmeas
     
def plot_convergence(Energies,coeffs):
    fig = plt.figure(figsize = (8,6 ))
    for c in coeffs:
        plt.plot(Energies[c], label = c, color = cm.gnuplot(c), linewidth = 2)
        plt.xscale('log')
    plt.legend(loc = 3)
    plt.xlabel('# iteration')
    plt.ylabel('Energy/KT')
    plt.figtext(0.99, 0.99, git_v, fontsize = 8, ha = 'right', va = 'top')
    plt.savefig('D:/Google Drive/Potential Retrieval/convergence_%dmcs.png' % N_mcs, dpi = 600)
    plt.close('all')

    return
    
    
def plot_conf(particles,N_mcs,n_conf):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(particles[:,0],particles[:,1],particles[:,2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.figtext(0.99, 0.99, git_v, fontsize = 8, ha = 'right', va = 'top')
    plt.savefig('D:/Google Drive/Potential Retrieval/final_configuration/sim_%dmcs_n%d.png' % (N_mcs,n_conf), dpi = 600)
    plt.close('all')
    
    return     
    #%%
def check_correlation_at_convergence(E_conv, N_display = 20000):

    plt.plot(range(N_conv,N_conv+N_display),E_conv[:N_display])
    plt.xlabel('# MCstep')
    plt.ylabel('Energy')
    plt.figtext(0.99, 0.99, git_v, fontsize = 8, ha = 'right', va = 'top')
    plt.savefig('D:/Google Drive/Potential Retrieval/after_convergence_Nmcs%d_dr%.2f.png' % (N_mcs,dr_c), dpi = 600)
    plt.close('all')
    
    interval = N_mcs/5
    shift_lim = 10000
    center = len(E_conv)/2
    sample = E_conv[center-interval/2:center+interval/2]
    corr = []
    shift = []
    for i in range(center-interval/2 - shift_lim, center-interval/2 + shift_lim):
        c = np.corrcoef(E_conv[i:interval+i],sample)
        corr.append(c[0,1])
        shift.append(i - center + interval/2)
    
    minorLocator = MultipleLocator(1000)
    majorLocator = MultipleLocator(5000)
    majorFormatter = FormatStrFormatter('%d')
    fig,ax = plt.subplots()
    ax.plot(shift,corr)
    ax.set_xlabel(r'$dt$')
    ax.set_ylabel(r'$\langle\,E(t)\,E(t+dt)\,\rangle$')
    ax.xaxis.set_major_locator(majorLocator)
    ax.xaxis.set_major_formatter(majorFormatter)
    ax.xaxis.set_minor_locator(minorLocator)
    plt.figtext(0.99, 0.99, git_v, fontsize = 8, ha = 'right', va = 'top')
    plt.savefig('D:/Google Drive/Potential Retrieval/after_convergence_correlation_Nmcs%d_dr%.2f.png' % (N_mcs,dr_c), dpi = 600)
    plt.close('all')

    return
    #%%
def calc_g_average(g_list):
    Nm = len(g_list)
    r = g_list[0].r
    gsum2 = np.zeros(len(r))
    gsum = np.zeros(len(r))
    for gm in g_list:
        gsum += gm.v
        gsum2 += gm.v**2
    gav = gsum/Nm
    gstd = np.sqrt(gsum2/Nm - gav**2) * np.sqrt(Nm) / np.sqrt(Nm-1.0)  

    np.savetxt('D:/Google Drive/Potential Retrieval/final_g/g_av_%dmcs_conv%d_skip%d.txt' % (N_mcs,N_conv,N_corr),np.transpose([r,gav,gstd]), fmt = '%.04f', delimiter = '\t', header = 'r\tg(r)\tsigma(g)')
    
    plt.figure(figsize = (6,4.5))
    plt.plot(gtheory.r,gtheory.v,label = 'expected', zorder = 0)
    plt.errorbar(r,gav,yerr = gstd,marker = 'o', linestyle = 'None', label = 'measured', zorder = 1)
    plt.xlabel(r'$r$')
    plt.ylabel(r'$g(r)$')
    plt.figtext(0.99, 0.99, git_v, fontsize = 8, ha = 'right', va = 'top')
    plt.legend()
    plt.savefig('D:/Google Drive/Potential Retrieval/final_g/g_av_%dmcs_conv%d_skip%d.png' % (N_mcs,N_conv,N_corr), dpi = 300)
    plt.close('all')
    
    return
    
    #%%
            
if __name__ == '__main__':

    
    git_v = subprocess.check_output(["git", "rev-parse", "--verify", "--short", "HEAD"])
    git_v = git_v.strip().decode('UTF-8')

    gtheory = par()
    gtheory.r,gtheory.v = get_g('D:/Google Drive/Potential Retrieval/gtest.txt')
    N_mcs = 80000
    dr_c = 0.58
    L_box = 20
    N_particles = 50
    dim = 3    
    # Monte Carlo Step at which I have convergence
    N_conv = 50000
    # MC steps to wait between saving observable
    N_corr = 2000
    
    check_folders_existence('D:/Google Drive/Potential Retrieval/')

    
    # Define a potential
    vtest = par()
    vtest.r,vtest.v = get_g('D:/Google Drive/Potential Retrieval/vtest.txt')
#    vtest.r,vtest.v = gtheory.r,-np.log(gtheory.v)
#    vtest.v[0] = 100000
    vtest.bin = np.append(0,np.append(0.5*(vtest.r[1:]+vtest.r[:-1]),2*L_box))


    # If I want to try different dr coefficients
    coeffs = [dr_c]
    Energies = {}

    for i,c in enumerate(coeffs):
        

        #%%
        particles,E,g_list = run_montecarlo(vtest, i, dr_coeff = c)
        Energies[c] = E
        
#%% Plot the convergence test
    plot_convergence(Energies,coeffs)
    
#%% Correlate after convergence
#    check_correlation_at_convergence(Energies[dr_c][N_conv:])

#%% Save the statistical average of g
    calc_g_average(g_list)
    

#%%

    

#    g_arr = np.array(g_list)
#    g_av = np.average(g_list, axis = 0)
#    g_std = np.std(g_list, axis = 0)
#    plt.errorbar(g.r,g_av,yerr = g_std, color = 'red')
#    plt.plot(g.r,g.v, color = 'green')
#
#
#
#    end = time.time()
#    duration = end - start
#    print('Done in %.1f s' % duration)
