import scipy as spi 
import numpy as np 

identity = np.array([[1,0], [0,1]])
pauli_x = np.array([[0, 1], [1, 0]])
pauli_y = np.array([[0, -1j], [1j, 0]])
pauli_z = np.array([[1, 0], [0, -1]])

def ket_to_density(ket):
    '''
    Takes a ket vector and returns the corresponding density matrix
    '''
    return np.outer(ket, ket.conj())

def bloch_vector_to_density(r):
    '''
    Takes a Bloch vector and returns the corresponding density matrix
    '''
    return 0.5 * (np.eye(2) + r[0] * np.array([[0, 1], [1, 0]]) +
                  r[1] * np.array([[0, -1j], [1j, 0]]) +
                  r[2] * np.array([[1, 0], [0, -1]]))

def solver(rho_0, evolution, t_span=[0, 10]):
    '''
    Solves the system by veccing the initial state and feeding it into the RK45 solver
    '''
    rho_size = rho_0.size

    vectorized_rho = rho_0.reshape(rho_size)
    
    solution = spi.solve_ivp(evolution, t_span, vectorized_rho, method='RK45', t_eval=np.linspace(0, 10, 100))
    
    return solution

def master_derivative(rho, H, jump_operators):
    '''
    Computes the derivative of the density matrix rho given the Hamiltonian H and the list of jump operators
    '''
    # Compute Hamiltonian part:
    hamiltonian_part = -1j * (np.dot(H, rho) - np.dot(rho, H))
    jump_part = 0

    # Compute non-Hamiltonian part:
    for jump_operator in jump_operators:
        jump_part += (np.dot(np.dot(jump_operator, rho), jump_operator.conj().T)
                      - 0.5 * np.dot(jump_operator.conj().T, np.dot(jump_operator, rho))
                      - 0.5 * np.dot(rho, np.dot(jump_operator.conj().T, jump_operator)))

    # Return the sum of the two parts:
    return hamiltonian_part + jump_part

def evolution_to_feed_rk4(t, vec, f, rho_dims):
    '''
    Reshapes the vector into a matrix, computes the derivative and reshapes it back into a vector.
    This way it can be refed into the RK45 solver.
    '''
    # Shape into matrix
    rho = vec.reshape(rho_dims)
    # Evolve using master equation
    rho = f(rho)
    # Reshape into vector
    return rho.reshape(rho.size)

def Channel(rho : np.array, Hamilt : np.array) : 
    return np.dot(Hamilt, np.dot(rho, Hamilt.conj().T))

def R(Lambda : np.array) : 
    TransferMatrix = np.zeros((4,4),dtype=complex)
    sigmas = [identity, pauli_x, pauli_y, pauli_z]
    for i in range(0,3) : 
        for j in range(0,3) : 
            TransferMatrix[i,j] = (1/2)* np.trace(np.dot(sigmas[i], Channel(sigmas[j],Lambda)))
    return TransferMatrix

if __name__ == "__main__" : 
    # Initial state
    r_0 = np.array([1, 1, 5])
    r_0 = r_0 / np.linalg.norm(r_0)
    rho_0 = bloch_vector_to_density(r_0)
    
    # Evolution, please define also the correct jump rates
    time = np.pi/2
    Hamiltonian = (1/np.sqrt(2))*np.array([[1, 1], [1, -1]])  # Evolution is a simple rotation around the z-axis
    dephasing_rate = 0
    Jump_list = [np.sqrt(dephasing_rate) * np.array([[1, 0], [0, -1]])]

    print('\nHamiltonian:\n', Hamiltonian)
    
    #Quantum process tomography part
    print(R(Channel(rho_0, Hamiltonian)))