import configparser
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import subprocess
import os
import sys
import itertools
import copy

#%%
class par:
    def __init__(self):
        self.r = 0
        self.v = 0
        self.bin = 0

def check_folders_existence(f_path):    
    
    folders_list = ['', 'iters_output']

    if PR_par['Plot all g']: # If I need to plot all the g(r) prepare folders
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

def pair_correlation_function_2D(x, y, sign, S, rMax, dr, kind = 'unsigned'): 
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
            elif(kind == 'same'):
                numberDensity = len(d_pos) / np.double(S**2)
                g[p, :] = pos_result/numberDensity
            elif(kind == 'opp'):
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
#%%    
def replicate_particles(particles):
    shifts = list(itertools.product([0,1,2], repeat = MC_par['dim']))
    more_particles = np.empty((0,MC_par['dim'] + 1 * MC_par['charge']))
    for s in shifts:
        delta = MC_par['L_box']*np.array(s)
        if MC_par['charge']:
            delta = np.append(delta,0)
        more_particles = np.append(more_particles,particles + delta,axis = 0)
    return more_particles
#%%
def replicate_particles_cut(particles, r_cut):
    
    more_particles_cut = replicate_particles(particles)
    ctr = 1.5 * MC_par['L_box']
    r_lim = r_cut + 0.5 * MC_par['L_box']
    
    for i in range(MC_par['dim']):  # cut away particles further than lim from ctr
        more_particles_cut = more_particles_cut[np.abs(more_particles_cut[:,i] - ctr) < r_lim]

    for i in range(MC_par['dim']):
        more_particles_cut[:,i] -= (MC_par['L_box'] - r_cut)
    
    return more_particles_cut

#%%
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

#%%
def initialize_system(how):
    """Initialize an array of positions of N particles in a box of size L"""
    
    if how == 'random':
        particles = np.random.rand(MC_par['N_particles'],MC_par['dim'])*MC_par['L_box']         
    elif 'array' in how:
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
        
        if 'noisy' in how:
            noise = (np.random.rand(MC_par['N_particles'],MC_par['dim']) - 0.5) * 0.5 * MC_par['L_box']/n
            particles = particles + noise
            
        if 'charged' in how:
            particles = np.append(particles, np.ones((MC_par['N_particles'],1)), axis = 1) # add a column for charge
            # and flip half charges
            particles[::2,2] *= -1
                
    return particles
#%%
def lj(r):
    """Lennard-Jones potential"""
    sigma = 5.
    eps = 2.    
    
    return 4*eps*((sigma/r)**12-(sigma/r)**6)   
  
def calc_distances(elements,ref,R_cut):
    #%%
    starting_shape = np.shape(elements)
    #%% Calculate the dx and dy in the next unit cell (+- L_box)
    delta2 = np.reshape((elements - ref)**2,-1)
    delta2_p = np.reshape((elements - ref + MC_par['L_box'])**2,-1)
    delta2_m = np.reshape((elements - ref - MC_par['L_box'])**2,-1)
    #%% And take the distance that is minimum in each direction
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
    new_particle = copy.copy(old_particle)  # otherwise I move it anyway
    new_particle[:MC_par['dim']] = (old_particle[:MC_par['dim']] + dr) % MC_par['L_box']
    #%% Apply periodic Boundary conditions and exclude particles outside R_cut
    # Particles at a distance > R_cut don't contribute to the energy
    dE = 0
    #%%
    if(MC_par['charge']):
        for charge_prod in ['same','opp']:
            #%%
            sel_other_particles = other_particles[other_particles[:,-1] * old_particle[-1] == word2sign[charge_prod]]
            old_distances = calc_distances(sel_other_particles[:,:MC_par['dim']],old_particle[:MC_par['dim']],R_cut)   
            new_distances = calc_distances(sel_other_particles[:,:MC_par['dim']],new_particle[:MC_par['dim']],R_cut)
            old_histo,bins  = np.histogram(old_distances, bins = v_bin)
            new_histo,bins  = np.histogram(new_distances, bins = v_bin)
            dE += np.sum( (new_histo-old_histo) * v[charge_prod] )
            #%%
    else:
        old_distances = calc_distances(other_particles,old_particle,R_cut)   
        new_distances = calc_distances(other_particles,new_particle,R_cut)
        old_histo,bins  = np.histogram(old_distances, bins = v_bin)
        new_histo,bins  = np.histogram(new_distances, bins = v_bin)
        dE += np.sum((new_histo-old_histo)*v['unsigned'])
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
#%%
    R_cut = g_r_max  # TODO
    #v_f = interp1d(r,v)

    g_lists = {'same':[],'opp':[]} 
    S_lists = {'same':[],'opp':[]}
    
    # Initialize the system at a random distribution
    particles = initialize_system(MC_par['init_conf'])
    E = np.zeros(MC_par['N_mcs']+2) #TODO initial energy
    MC_move = 0
#%%
    for n in range(MC_par['N_mcs']+1):
        #%%
        chosen_one = np.random.randint(MC_par['N_particles'])
        dr = dr_coeff * ( np.random.rand(MC_par['dim']) - 0.5 )
        #%%###
        dE,this_move = MC_step(particles,chosen_one,dr,R_cut,v)
        ####
        E[n+1] = E[n] + dE
        MC_move += this_move
        
        ## Every N_corr steps
        if(n % MC_par['N_corr']==0):
            print_progress(n+1,MC_par['N_mcs'])    
            ## After convergence calc and save g
            if(n > MC_par['N_conv']):
                for kind in ['same','opp']:
                    g_meas, S_meas, r_meas_Sg = calc_and_plot_g_r(particles,n,iteration,kind = kind, save_plot = PR_par['Plot all g'])
                    g_lists[kind].append(g_meas)
                    S_lists[kind].append(S_meas)
            
    elapsed = time.time() - start
    print(' Done in %d s' % elapsed)
    
    print('%d %% of the Monte Carlo steps were performed (%d out of %d)' % (100.0*MC_move/MC_par['N_mcs'], MC_move,MC_par['N_mcs']))

    plot_conf(particles,iteration)

        
    return particles,E,g_lists,S_lists,r_meas_Sg
    
def calc_and_plot_g_r(particles,n,iteration, kind = 'unsigned', save_plot = False):
    
    #If needed replicate the system that I considered periodic
    if PR_par['Replicate Particles']:
        more_particles = replicate_particles_cut(particles, g_r_max)
        S = MC_par['L_box'] + 2 * g_r_max
    else:
        more_particles = particles
        S = MC_par['L_box']
        
    if MC_par['dim'] == 3:
        if MC_par['charge']:
            [xp,yp,zp,sp] = np.transpose(more_particles)
            # Calculate pair correlation function
            g_meas,S_meas,r_meas_Sg,_ = pair_correlation_function_3D(xp,yp,zp,S,9.75,0.5,kind)
        else:
            [xp,yp,zp] = np.transpose(more_particles)
            # Calculate pair correlation function
            g_meas,S_meas,r_meas_Sg,_ = pair_correlation_function_3D(xp,yp,zp,S,9.75,0.5,kind)
        
    elif MC_par['dim'] == 2:
        if MC_par['charge']:
            [xp,yp,sp] = np.transpose(more_particles)
            g_meas,S_meas,r_meas_Sg,_ = pair_correlation_function_2D(xp,yp,sp,S,g_r_max,g_dr,kind)
        else:
            [xp,yp] = np.transpose(more_particles)
            g_meas,S_meas,r_meas_Sg,_ = pair_correlation_function_2D(xp,yp,S,g_r_max,g_dr,kind)
    if save_plot:
        plt.figure(figsize = (7,4))
        plt.plot(r_meas_Sg,g_meas)
        plt.plot(g_th_r,g_th[kind])
        plt.xlabel('r')
        plt.ylabel('g(r)')
        plt.figtext(0.99, 0.99, git_v, fontsize = 8, ha = 'right', va = 'top')
        plt.savefig(out_root + 'iters_output/all_g_%03d/%s_%d.png' % (iteration,kind,n), dpi = 300)
        plt.close('all')
    
    return g_meas, S_meas, r_meas_Sg
 #%%    
def plot_convergence(Energies,coeffs,iteration):
    xscale = int(np.log10(MC_par['N_mcs'])) + 4
    plt.figure(figsize = (xscale,6))
    cmax = np.max(coeffs)
    for c in coeffs:
        plt.plot(Energies[c], label = c, color = cm.gnuplot(c/cmax), linewidth = 2)
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
    if MC_par['charge']:
        charge = particles[:,-1]
    else:
        charge = np.zeros(len(particles))
        
    if MC_par['dim'] == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(particles[:,0],particles[:,1],particles[:,2], c = charge, alpha = 0.75)
        ax.set_zlabel('z')
    elif MC_par['dim'] == 2:
        ax = fig.add_subplot(111)
        ax.scatter(particles[:,0],particles[:,1], c = charge, alpha = 0.75)        
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.figtext(0.99, 0.99, git_v, fontsize = 8, ha = 'right', va = 'top')
    plt.savefig(out_root + 'iters_output/final_conf_%03d.png' % (iteration), dpi = 600)
    plt.close('all')
    
    return     
    #%%
def check_correlation_at_convergence(E_conv, N_display = 20000):

    plt.plot(np.arange(MC_par['N_conv'],MC_par['N_conv']+N_display),E_conv[:N_display])
    plt.xlabel('# MCstep')
    plt.ylabel('Energy')
    plt.figtext(0.99, 0.99, git_v, fontsize = 8, ha = 'right', va = 'top')
    plt.savefig(out_root + 'after_convergence_Nmcs%d_dr%.2f.png' % (MC_par['N_mcs'],MC_par['dr_c']), dpi = 600)
    plt.close('all')
    
    hinterval = int(MC_par['N_mcs']/5.0/2.0)
    shift_lim = 10000
    center = int(len(E_conv)/2.0)
    sample = E_conv[center-hinterval:center+hinterval]
    corr = []
    shift = []
    for i in range(center-hinterval - shift_lim, center - hinterval + shift_lim):
        c = np.corrcoef(E_conv[i:2*hinterval+i],sample)
        corr.append(c[0,1])
        shift.append(i - center + hinterval)
    
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
    what,kind = name.split('_')
    if what == 'g':
        plt.plot(g_th_r,g_th[kind],label = 'expected', zorder = 0)
    elif what == 'S':
        plt.plot(g_th_r, S_th[kind], label = 'expected', zorder = 0)
    plt.errorbar(r,g_av,yerr = g_std, marker = 'o', markersize = 3, linestyle = 'None', label = 'measured', zorder = 1)
    plt.xlabel(r'$r$')
    plt.ylabel(r'$%s_{%s}(r)$' % (what,kind))
    plt.xlim([0,g_r_max])
    plt.figtext(0.99, 0.99, git_v, fontsize = 8, ha = 'right', va = 'top')
    plt.legend()
    plt.savefig(out_root + 'iters_output/%s_av_%03d.png' % (name,iteration), dpi = 300)
    plt.close('all')
    
    return g_av,g_std
    
    #%%

def straighten_pot(fr,f):

    minima = m = np.r_[False,f[1:] < f[:-1]] & np.r_[f[:-1] < f[1:], False]

    first_minimum = np.where(m)[0][0]
    r_temp = np.append(fr[:first_minimum],fr[minima])
    r_temp = np.append(r_temp,20)
    f_temp = np.append(f[:first_minimum],f[minima])
    f_temp = np.append(f_temp,0)
    f_func = interp1d(r_temp,f_temp, kind = 'linear')
    
    new_f = f_func(fr)
    
    return new_f
    #%%
def save_and_plot_new_potential(new_v, kind, v_list):
    
    np.savetxt(out_root + 'iters_output/pot_' + kind + '%03d.txt' % (k+1),np.transpose([v_r,new_v[kind]]), fmt = '%.04f', delimiter = '\t', header = 'r\t\tV(r)')
            
            
    # And plot it with the others (max 10 others)
    to_skip = int(len(v_list)/10) + 1
    plt.figure(figsize = (6,4.5))
    for i,vv in enumerate(v_list[::to_skip]):
        plt.plot(v_r[1:],vv[kind][1:], label = 'iteration #%03d' % (i*to_skip))
        plt.xlabel(r'$r$')
        plt.ylabel(r'$V_{%s}(r)$' % kind)
    plt.xlim([0,g_r_max])
    plt.legend()    
    plt.savefig(out_root + 'iters_output/pot_%s_000-%03d.png' % (kind,k+1),dpi = 300)
    plt.close('all')
    
    return
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
def plot_pot(pot_r,potentials):
    
    plt.figure
    for this_pot in potentials:
        plt.plot(pot_r,potentials[this_pot], label = this_pot)
    plt.ylim([-1,3])
    plt.legend()
    plt.xlabel(r'$r$')
    plt.ylabel(r'$V(r)$')
    plt.savefig(out_root + 'starting_pot.png', dpi = 300)
    plt.close('all')
    
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
    MC_par['N_mcs'] = int(1.5e+6)
    MC_par['L_box'] = 10
    MC_par['N_particles'] = int(MC_par['L_box']**2 * np.pi) # wavelength = 1, density of berrydennis
    MC_par['dim'] = 2  
    MC_par['dr_c'] = MC_par['L_box'] / np.power(MC_par['N_particles'],1.0/MC_par['dim'])
    # Monte Carlo Step at which I have convergence
    MC_par['N_conv'] = 1e+5
    # MC steps to wait between saving observable
    MC_par['N_corr'] = 1e+4
    # Initialization of the particles in the box
    MC_par['init_conf'] = 'noisy_charged_array'
    MC_par['charge'] = True
    MC_par['check correlation at convergence'] = True
    word2sign = {'same':1,'opp':-1}
    
    PR_par = {}
    # Number of iterations of Potential retrieval alghoritm
    PR_par['N_iter'] = 1
    PR_par['damping'] = 0.02
    PR_par['g_name'] = 'BD_sameopp_6_002'
    PR_par['Replicate Particles'] = True
    PR_par['Plot all g'] = False
    PR_par['Zero potential from'] = 2.0

    sim_details = '%s/L=%d_%.1EMCS_%03dITER/' % (PR_par['g_name'],MC_par['L_box'],MC_par['N_mcs'],PR_par['N_iter'])
    out_root = root_dir + sim_details
    check_folders_existence(out_root)
    print('Working on %s' % sim_details[:-1])

    init_conf_file()


    # Get the g(r)
    g_th, S_th = {}, {}
    g_th_r,g_th['same'],g_th['opp'] = np.loadtxt(root_dir + 'g_%s.txt' % PR_par['g_name'], dtype = 'float', unpack = 'true')
    g_th['unsigned'] = (g_th['same'] + g_th['opp']) * 0.5
    g_dr = g_th_r[1] - g_th_r[0]
    if PR_par['Replicate Particles']:
        g_r_max = np.min([g_th_r[-1],MC_par['L_box'] / 2.0])
    else:
        g_r_max = np.min([g_th_r[-1],MC_par['L_box'] / 4.0])
    if MC_par['dim'] == 2:
        S_th['same'] = g_th['same'] * 2 * np.pi * g_th_r
        S_th['opp'] = g_th['opp'] * 2 * np.pi * g_th_r
        S_th['unsigned'] = g_th['unsigned'] * 2 * np.pi * g_th_r
    elif MC_par['dim'] == 3:
        S_th['same'] = g_th['same'] * 4 * np.pi * g_th_r**2
        S_th['opp'] = g_th['opp'] * 4 * np.pi * g_th_r**2
        S_th['unsigned'] = g_th['unsigned'] * 4 * np.pi * g_th_r**2
    
    # Define a potential
#    v_r,v_trial = get_g(root_dir + 'vtest.txt')
    v_r = g_th_r
    v_bin = np.append(0,np.append(0.5*(v_r[1:]+v_r[:-1]),10*MC_par['L_box']))
    v_trial = {}
    g_th['same'][g_th['same'] <=0] = 1e-10 # to account for the infinity at the beginning
    v_trial['same'] = - np.log(g_th['same'])
    v_trial['opp'] = - np.log(g_th['opp'])
    for kind in v_trial: # decrease coherence
        v_trial[kind][v_r > PR_par['Zero potential from']] = 0

#    for kind in v_trial:
#        v_trial[kind] = straighten_pot(v_r,v_trial[kind])
#    v_trial = np.append(v_trial[:100],v_trial[100:]/v_r[100:]) 
    plot_pot(v_r,v_trial)


    # If I want to try different dr coefficients
    coeffs = np.array([0.5]) * MC_par['dr_c']
    v_list = []
    v_list.append(v_trial)
    
#%%    
    for k in range(PR_par['N_iter']):

        Energies = {}
        
        print('Iteration #%d' % (k + 1))        

        for i,c in enumerate(coeffs):
            
    
            #%%
            particles,E,g_lists,S_lists,g_meas_r = run_montecarlo(v_list[k], k+1, i, dr_coeff = c)
            Energies[c] = E
            
    #%% Plot the convergence test
        if MC_par['check correlation at convergence']:
            plot_convergence(Energies,coeffs, k+1)
        
    #%% Correlate after convergence
        check_correlation_at_convergence(Energies[c][int(MC_par['N_conv']):], N_display = 500 * MC_par['L_box']**MC_par['dim'])
    
    #%% Save the statistical average of g
        for kind in g_lists:    
            gav,gstd = calc_dist_average(g_lists[kind],g_meas_r,'g_' + kind,k+1)
            Sav,Sstd = calc_dist_average(S_lists[kind],g_meas_r,'S_' + kind,k+1)
        
    #%% Define what part of the calc g can be compared to the known one
        r_min = np.where(np.abs(g_meas_r[0] - g_th_r) < 1e-4)[0][0]
        r_max = np.where(np.abs(g_meas_r[-1] - g_th_r) < 1e-4)[0][0] + 1
    #%% Perform the retrieval alghorithm
        new_v = {}
        for kind in ['same','opp']:       
            S_array = np.array([single_S[r_min:r_max] for single_S in S_lists[kind]])
            S_av = np.average(S_array, axis = 0)
            S_cov = np.cov(S_array,rowvar = 0)
        
            delta_S = S_av - S_th[kind][r_min:r_max]
                
            for nskip in range(0,8):
                try:
                    delta_v = np.linalg.solve(S_cov[nskip:,nskip:],delta_S[nskip:]) * PR_par['damping']
                    break
                except:
                    continue
            
            new_v[kind] = copy.copy(v_list[k][kind])
            new_v[kind][r_min + nskip:r_max] += delta_v
            
                    
        v_list.append(new_v)
        
        
        for kind in ['same','opp']:
            
            # Save the new potential and plot it with the others (max 10 others)
            save_and_plot_new_potential(new_v, kind, v_list)
            
        #%%            

