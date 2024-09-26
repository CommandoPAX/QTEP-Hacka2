import numpy as np
import scipy.linalg as la

def ket_to_density(ket) :
    return np.outer(ket, ket.conj())

def rho_derivative(rho, H, L) :
    # Takes as arguments the Hamiltonian H and the list of jump operators L

    # Compute Hamiltonian part :
    hamiltonian_part = -1j*(np.dot(H,rho) - np.dot(rho,H))
    linblad_part = 0

    # Compute non-Hamiltonian part :
    for jump_operator in L:
        linblad_part += np.dot(np.dot(jump_operator,rho),jump_operator.conj().T) - 0.5*np.dot(jump_operator.conj().T,jump_operator,rho) - 0.5*np.dot(rho,jump_operator.conj().T,jump_operator)

    # Return the sum of the two parts :
    return hamiltonian_part + linblad_part




x = np.array([0,1])
print(ket_to_density(x))