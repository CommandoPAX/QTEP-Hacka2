import numpy as np
import scipy.integrate as spi

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

def RungeKutta(x, y, dx, dydx):
    
    # Calculate slopes
    k1 = dx*dydx(x, y)
    k2 = dx*dydx(x+dx/2., y+k1/2.)
    k3 = dx*dydx(x+dx/2., y+k2/2.)
    k4 = dx*dydx(x+dx, y+k3)
    
    # Calculate new x and y
    y = y + 1./6*(k1+2*k2+2*k3+k4)
    x = x + dx
    
    return x, y

def solver(rho,f) :
    rho_size = rho.size

    vectorized_rho = rho.reshape(rho_size,1)
    
    result = RungeKutta(f,0,vectorized_rho,1)
    
    return result

def evolution(vec,f,rho_dims) :
    # Shape into matrix
    rho = vec.reshape(rho_dims)
    # Evolve using master equation
    rho = f(rho)
    # Reshape into vector
    rho = rho.reshape(rho.size,1)
    return rho

# Initial state
ket_init = np.array([1,1])/np.sqrt(2)
rho_init = np.outer(ket_init,ket_init.conj())

# Evolution
Hamiltonian = np.array([[1,0],[0,-1]])
Jump_list = []

f = lambda rho : rho_derivative(rho,Hamiltonian,Jump_list)

result = solver(rho_init,f)
print(type(result))
