# QTEP-Hacka2

## Error process considered 
- Qubit dephasing
- Population leakage (maybe)


import scipy 
import numpy as np

def SuperLindblad(rho : np.array, SponEmiss, DephasRate) : 
    sigma_m = np.array([0,0],[1,0])
    sigma_p = np.array([0,1],[0,0])
    sigma_z = np.array([1,0], [0,-1])

    return (SponEmiss/2)*(2*np.dot(sigma_m, np.dot(rho, sigma_p)) - np.dot(sigma_p, np.dot(sigma_m, rho)) - np.dot(rho, np.dot(sigma_p,  sigma_m)) + (DephasRate/2)*(np.dot(sigma_z, np.dot(rho, sigma_z) - rho)

def evol (rho : np.array, Lindblad : np.array) : 
