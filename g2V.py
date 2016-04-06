import numpy as np
import matplotlib.pyplot as plt

def get_g(file_name):
    """Gets the radial distribution function from file_name"""
    
    r,g = np.loadtxt(file_name, dtype = 'float', unpack = 'true')
    
    return r,g
    
def init_V(g):
    """Initialize the potential with the effective potential function approximation"""
    return -np.log(g)
    
    
def MC_sim:
    """Performs a Monte Carlo simulation"""

    # Initialize the system

    for i in range(N_iterations):    
        #Propose a movement
    
        #Calculate the difference in energy

        #Accept or decline the movement

    return particles