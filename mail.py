import numpy as np
import scipy.linalg as la

def ket_to_density(ket) :
    return np.outer(ket, ket.conj())

def rho_derivative(rho, H, SponEmiss = 1, PureDephase = 1) :
    # Takes as arguments the Hamiltonian H and the list of jump operators L
    sigma_m = np.array([[0,0],[1,0]]) 
    sigma_p = np.array([[0,1],[0,0]]) 
    sigma_z = np.array([[1,0], [0,-1]])

    # Compute Hamiltonian part :
    hamiltonian_part = -1j*(np.dot(H,rho) - np.dot(rho,H))
    linblad_part = 0

    # Compute non-Hamiltonian part :
    
    linblad_part += (SponEmiss/2)*(2*np.dot(np.dot(sigma_m, rho), sigma_p) - np.dot(np.dot(sigma_p, sigma_m), rho) - np.dot(np.dot(rho, sigma_p),  sigma_m)) + (PureDephase/2)*(np.dot(np.dot(sigma_z, rho), sigma_z) - rho)
    # Return the sum of the two parts :
    return hamiltonian_part + linblad_part

sigma_z = [np.array([[1,0],[0,-1]])]
x = np.array([1,1])/np.sqrt(2)
H = np.array([[0,0],[0,0]])
print(rho_derivative(ket_to_density(x),H, 0, 1))
