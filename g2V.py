import configparser
import numpy as np
import matplotlib.pyplot as plt
#from scipy.interpolate import interp1d
import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import subprocess
import os
import sys
import itertools


class par:
    def __init__(self):
        self.r = 0
        self.v = 0
        self.bin = 0

def check_folders_existence(f_path):    
    
    folders_list = ['', 'iters_output']
    for k in range(PR_par['N_iter']):
        folders_list.append('iters_output/all_g_%03d' % (k+1))


    for folder in folders_list:        
        if not os.path.exists(f_path + folder):
            os.makedirs(f_path + folder)
            
    return


def print_progress(done,total):
    """print a progress bar"""
    
    percent = 100.0*done/(total)    
    bar = int(0.2*percent)    
    
    sys.stdout.write('\r')
    sys.stdout.write('[%-20s] %d%%' % ('='*bar, percent))
    sys.stdout.flush()
    
    return

def pair_correlation_function_2D(x, y, S, rMax, dr, kind = 'unsigned'): 
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
    S_average = np.zeros(num_increments)
    for i in range(num_increments): 
        radii[i] = (edges[i] + edges[i+1]) / 2. 
        rOuter = edges[i + 1] 
        rInner = edges[i]
        S_average[i] = np.mean(g[:, i]) / (rOuter - rInner)
        g_average[i] = np.mean(g[:, i]) / (np.pi * (rOuter**2 - rInner**2)) 
	
    return (g_average, S_average, radii, interior_indices)    
    
def replicate_particles(particles):
    shifts = list(itertools.product([0,1,2], repeat = MC_par['dim']))
    more_particles = np.empty((0,MC_par['dim']))
    for delta in shifts:
        more_particles = np.append(more_particles,particles + MC_par['L_box']*np.array(delta),axis = 0)
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
    g_average = np.zeros(num_increments)
    S_average = np.zeros(num_increments)
    for i in range(num_increments):
        radii[i] = (edges[i] + edges[i+1]) / 2.
        rOuter = edges[i + 1]
        rInner = edges[i]
        S_average[i] = np.mean(g[:, i]) / (rOuter - rInner)
        g_average[i] = np.mean(g[:, i]) / (4.0 / 3.0 * pi * (rOuter**3 - rInner**3))

    return (g_average, S_average, radii, interior_indices)
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

def initialize_system(how):
    """Initialize an array of positions of N particles in a box of size L"""
    
    if how == 'random':
        particles = np.random.rand(MC_par['N_particles'],MC_par['dim'])*MC_par['L_box']         
    elif how == 'array' or MC_par['init_conf']:
        n = np.power(MC_par['N_particles'],1.0/MC_par['dim'])
        n = int(n) + 1
        n_generated = n**MC_par['dim']
        if MC_par['dim'] == 2:
            X,Y = np.mgrid[0:n,0:n]
            more_particles = np.array([X.flatten(),Y.flatten()]).T
        elif MC_par['dim'] == 3:
            X,Y,Z = np.mgrid[0:n,0:n,0:n]
            more_particles = np.array([X.flatten(),Y.flatten(),Z.flatten()]).T
        n_excess = n_generated - MC_par['N_particles']
        # Remove the particles in excess (randomly)
        to_remove = np.random.permutation(n_generated)[:n_excess]
        particles = np.delete(more_particles, to_remove, axis = 0)
        # normalize
        particles = particles * MC_par['L_box'] / n
        
        if how == MC_par['init_conf']:
            noise = np.random.rand(MC_par['N_particles'],MC_par['dim']) - 0.5
            particles = particles + noise
            
                
    return particles

def lj(r):
    """Lennard-Jones potential"""
    sigma = 5.
    eps = 2.    
    
    return 4*eps*((sigma/r)**12-(sigma/r)**6)   
  
def calc_distances(elements,ref,R_cut):
    
    starting_shape = np.shape(elements)
    # Calculate the dx and dy in the next unit cell (+- L_box)
    delta2 = np.reshape((elements - ref)**2,-1)
    delta2_p = np.reshape((elements - ref + MC_par['L_box'])**2,-1)
    delta2_m = np.reshape((elements - ref - MC_par['L_box'])**2,-1)
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
    new_particle = (old_particle + dr)%MC_par['L_box']
    #%% Apply periodic Boundary conditions and exclude particles outside R_cut
    # Particles at a distance > R_cut don't contribute to the energy
    old_distances = calc_distances(other_particles,old_particle,R_cut)   
    new_distances = calc_distances(other_particles,new_particle,R_cut)
    old_histo,bins  = np.histogram(old_distances, bins = v_bin)
    new_histo,bins  = np.histogram(new_distances, bins = v_bin)
    dE = np.sum((new_histo-old_histo)*v)
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
            
def run_montecarlo(v,iteration , n_run, dr_coeff = 0.58):
    """ Test the MC simulation using the example in hte paper byLyubartsev and Laaksonen"""
    
    start = time.time()

    R_cut = v_r[-1]  # TODO
    #v_f = interp1d(r,v)

    g_list = [] 
    S_list = []
    
    # Initialize the system at a random distribution
    particles = initialize_system(MC_par['init_conf'])
    E = np.zeros(MC_par['N_mcs']+2) #TODO initial energy
    MC_move = 0

    for n in range(MC_par['N_mcs']+1):
        
        chosen_one = np.random.randint(MC_par['N_particles'])
        dr = dr_coeff*np.random.rand(MC_par['dim'])
        ####
        dE,this_move = MC_step(particles,chosen_one,dr,R_cut,v)
        ####
        E[n+1] = E[n] + dE
        MC_move += this_move
        
        ## Every N_corr steps
        if(n % MC_par['N_corr']==0):
            print_progress(n+1,MC_par['N_mcs'])    
            ## After convergence calc and save g
            if(n > MC_par['N_conv']):
                g_meas, S_meas, r_meas_Sg = calc_and_plot_g_r(particles,n,iteration)
                g_list.append(g_meas)
                S_list.append(S_meas)
        
            
    elapsed = time.time() - start
    print(' Done in %d s' % elapsed)
    
    print('%d %% of the Monte Carlo steps were performed (%d out of %d)' % (100.0*MC_move/MC_par['N_mcs'], MC_move,MC_par['N_mcs']))

    plot_conf(particles,iteration)

        
    return particles,E,g_list,S_list,r_meas_Sg
    
def calc_and_plot_g_r(particles,n,iteration, save_plot = False):
    
    #Replicate the system that I considered periodic
    more_particles = replicate_particles(particles)
    
    if MC_par['dim'] == 3:
        [xp,yp,zp] = np.transpose(more_particles)
        # Calculate pair correlation function
        g_meas,S_meas,r_meas_Sg,_ = pair_correlation_function_3D(xp,yp,zp,MC_par['L_box']*MC_par['dim'],9.75,0.5)
        
    elif MC_par['dim'] == 2:
        [xp,yp] = np.transpose(more_particles)
        g_meas,S_meas,r_meas_Sg,_ = pair_correlation_function_2D(xp,yp,MC_par['L_box']*MC_par['dim'],9.75,0.5)

    if save_plot:
        plt.figure(figsize = (7,4))
        plt.plot(r_meas_Sg,g_meas)
        plt.plot(g_th_r,g_th)
        plt.xlabel('r')
        plt.ylabel('g(r)')
        plt.figtext(0.99, 0.99, git_v, fontsize = 8, ha = 'right', va = 'top')
        plt.savefig(out_root + 'iters_output/all_g_%03d/mcs_%d.png' % (iteration,n), dpi = 300)
        plt.close('all')
    
    return g_meas, S_meas, r_meas_Sg
     
def plot_convergence(Energies,coeffs,iteration):
    xscale = int(np.log10(MC_par['N_mcs'])) + 4
    plt.figure(figsize = (xscale,6))
    for c in coeffs:
        plt.plot(Energies[c], label = c, color = cm.gnuplot(c), linewidth = 2)
        plt.xscale('log')
    plt.legend(loc = 3)
    plt.xlabel('# iteration')
    plt.ylabel('Energy/KT')
    plt.figtext(0.99, 0.99, git_v, fontsize = 8, ha = 'right', va = 'top')
    plt.savefig(out_root + 'iters_output/convergence_%03d.png' % iteration, dpi = 600)
    plt.close('all')

    return
    
#%%    
def plot_conf(particles,iteration):

    fig = plt.figure()
    if MC_par['dim'] == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(particles[:,0],particles[:,1],particles[:,2])
        ax.set_zlabel('z')
    elif MC_par['dim'] == 2:
        ax = fig.add_subplot(111)
        ax.scatter(particles[:,0],particles[:,1])        
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.figtext(0.99, 0.99, git_v, fontsize = 8, ha = 'right', va = 'top')
    plt.savefig(out_root + 'iters_output/final_conf_%03d.png' % (iteration), dpi = 600)
    plt.close('all')
    
    return     
    #%%
def check_correlation_at_convergence(E_conv, N_display = 20000):

    plt.plot(range(MC_par['N_conv'],MC_par['N_conv']+N_display),E_conv[:N_display])
    plt.xlabel('# MCstep')
    plt.ylabel('Energy')
    plt.figtext(0.99, 0.99, git_v, fontsize = 8, ha = 'right', va = 'top')
    plt.savefig(out_root + 'after_convergence_Nmcs%d_dr%.2f.png' % (MC_par['N_mcs'],MC_par['dr_c']), dpi = 600)
    plt.close('all')
    
    interval = MC_par['N_mcs']/5
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
    plt.savefig(out_root + 'after_convergence_correlation_Nmcs%d_dr%.2f.png' % (MC_par['N_mcs'],MC_par['dr_c']), dpi = 600)
    plt.close('all')

    return
    #%%
def calc_dist_average(g_list, r, name, iteration):
    
    N_av = len(g_list)
    g_array = np.array(g_list)
    g_av = np.average(g_array, axis = 0)
    g_std = np.std(g_array,axis = 0) / np.sqrt(N_av)    ## Error on the average
    np.savetxt(out_root + 'iters_output/%s_av_%03d.txt' % (name,iteration),np.transpose([r,g_av,g_std]), fmt = '%.5e', delimiter = '\t\t', header = 'r\tg(r)\tsigma(g)')
    
    plt.figure(figsize = (6,4.5))
    #Plot theory
    if name == 'g':
        plt.plot(g_th_r,g_th,label = 'expected', zorder = 0)
    elif name == 'S':
        plt.plot(g_th_r, g_th * 4 * np.pi * g_th_r**2,label = 'expected', zorder = 0)
    plt.errorbar(r,g_av,yerr = g_std, marker = 'o', markersize = 3, linestyle = 'None', label = 'measured', zorder = 1)
    plt.xlabel(r'$r$')
    plt.ylabel(r'$%s(r)$' % name)
    plt.figtext(0.99, 0.99, git_v, fontsize = 8, ha = 'right', va = 'top')
    plt.legend()
    plt.savefig(out_root + 'iters_output/%s_av_%03d.png' % (name,iteration), dpi = 300)
    plt.close('all')
    
    return g_av,g_std
    
    #%%
            
    
def init_conf_file():
    #%%
    config = configparser.ConfigParser()
    
    config['MC parameters'] = MC_par
    config['Potential Retrieval parameters'] = PR_par
    DT = time.localtime()
    config['General'] = {'date': '%02d/%02d/%d' % (DT.tm_mday, DT.tm_mon, DT.tm_year),
                          'time': '%d:%d' % (DT.tm_hour,DT.tm_min)}
    
    
    with open(out_root + 'info.cfg', 'w') as configfile:
        config.write(configfile)
#%%    
    return
    
    
    #%%
if __name__ == '__main__':

    root_dir = 'L:/NS/kuiperslab/Lorenzo/DATA/MC_SIM/'
    git_v = subprocess.check_output(["git", "rev-parse", "--verify", "--short", "HEAD"])
    git_v = git_v.strip().decode('UTF-8')
    

#    gtheory = par()
#    gtheory.r,gtheory.v = get_g(root_dir + 'g_paper.txt')
#    Stheory = par()
#    Stheory.r,Stheory.v = get_g(root_dir + 'g_paper.txt')
#    Stheory.v *= 4 * np.pi * Stheory.r**2
    MC_par = {}    #A dictionary for all MC parameters
    MC_par['N_mcs'] = int(1e+6)
    MC_par['dr_c'] = 0.58
    MC_par['L_box'] = 20
    MC_par['N_particles'] = 14
    MC_par['dim'] = 2  
    # Monte Carlo Step at which I have convergence
    MC_par['N_conv'] = 50000
    # MC steps to wait between saving observable
    MC_par['N_corr'] = 2500
    # Initialization of the particles in the box
    MC_par['init_conf'] = 'array_w_noise'
    
    PR_par = {}
    # Number of iterations of Potential retrieval alghoritm
    PR_par['N_iter'] = 13
    PR_par['damping'] = 2.0

    
    out_root = root_dir + '%dD_%.1EMCS_ITER%03d/' % (MC_par['dim'],MC_par['N_mcs'],PR_par['N_iter'])
    check_folders_existence(out_root)

    init_conf_file()


    # Get the g(r)
    g_th_r,g_th,_ = np.loadtxt(root_dir + 'gtest_%dD.txt' % MC_par['dim'], dtype = 'float', unpack = 'true')
    if MC_par['dim'] == 2:
        S_th = g_th * 2 * np.pi * g_th_r
    elif MC_par['dim'] == 3:
        S_th = g_th * 4 * np.pi * g_th_r**2
    
    # Define a potential
#    v_r,v_trial = get_g(root_dir + 'vtest.txt')
    v_r,v_trial = g_th_r, - np.log(g_th + (g_th == 0) * 1e-50) # to account for the infinity at the beginning
    v_bin = np.append(0,np.append(0.5*(v_r[1:]+v_r[:-1]),2*MC_par['L_box']))
    


    # If I want to try different dr coefficients
    coeffs = [MC_par['dr_c']]
    v_list = []
    v_list.append(v_trial)
    
#%%    
    for k in range(PR_par['N_iter']):

        Energies = {}
        
        print('Iteration #%d' % (k + 1))        

        for i,c in enumerate(coeffs):
            
    
            #%%
            particles,E,g_list,S_list,g_meas_r = run_montecarlo(v_list[k], k+1, i, dr_coeff = c)
            Energies[c] = E
            
    #%% Plot the convergence test
        plot_convergence(Energies,coeffs, k+1)
        
    #%% Correlate after convergence
    #    check_correlation_at_convergence(Energies[dr_c][N_conv:])
    
    #%% Save the statistical average of g
        gav,gstd = calc_dist_average(g_list,g_meas_r,'g',k+1)
        Sav,Sstd = calc_dist_average(S_list,g_meas_r,'S',k+1)
        
    #%% Define what part of the calc g can be compared to the known one
        r_min = np.where(g_meas_r == g_th_r[0])[0][0]
        r_max = np.where(g_meas_r == g_th_r[-1])[0][0] + 1
    #%% Perform the retrieval alghorithm
        
        S_array = np.array([single_S[r_min:r_max] for single_S in S_list])
        S_av = np.average(S_array, axis = 0)
        S_cov = np.cov(S_array,rowvar = 0)
        
        delta_S = S_av - S_th
                
        for nskip in range(1,5):
            try:
                delta_v = np.linalg.solve(S_cov[nskip:,nskip:],delta_S[nskip:]) * PR_par['damping']
                break
            except:
                continue
            
        new_v = np.zeros(len(v_r))
        new_v += v_list[k]
        new_v[nskip:] += delta_v
        v_list.append(new_v)
            
        # Save the new potential
        np.savetxt(out_root + 'iters_output/pot_%03d.txt' % (k+1),np.transpose([v_r,new_v]), fmt = '%.04f', delimiter = '\t', header = 'r\t\tV(r)')
        # And plot it with the others (max 10 others)
        to_skip = int(len(v_list)/10) + 1
        plt.figure(figsize = (6,4.5))
        for i,vv in enumerate(v_list[::to_skip]):
            plt.plot(v_r[1:],vv[1:], label = 'iteration #%03d' % (i*to_skip))
            plt.xlabel(r'$r$')
            plt.ylabel(r'$V(r)$')
        plt.legend()    
        plt.savefig(out_root + 'iters_output/pot_000-%03d.png' % (k+1),dpi = 300)
        plt.close('all')