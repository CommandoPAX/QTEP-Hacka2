import scipy.integrate as spi
import numpy as np 

identity = np.array([[1,0], [0,1]])
pauli_x = np.array([[0, 1], [1, 0]])
pauli_y = np.array([[0, -1j], [1j, 0]])
pauli_z = np.array([[1, 0], [0, -1]])
sigmas = [identity, pauli_x, pauli_y, pauli_z]

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
    
    solution = spi.solve_ivp(evolution, t_span, vectorized_rho, method='RK45', t_eval=np.linspace(t_span[0], t_span[1],100))
    
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

def state_tomography(operator : np.array) : 
    ExpectationVector = np.zeros((4),dtype=complex)
    sigmas = [identity, pauli_x, pauli_y, pauli_z]
    for i,sigma in enumerate(sigmas) :
            ExpectationVector[i] = np.trace(np.dot(sigma, operator))
    return ExpectationVector

def printArray(a):
    for row in range(len(a[0])):
        for col in range (len(a[0])):
            b = print("{:8.3f}".format(a[row][col]), end = " ")
        print(b)

if __name__ == "__main__" : 

    # Evolution, please define also the correct jump rates
    time = np.pi/2
    Hamiltonian = pauli_z  # Evolution is a simple rotation around the z-axis
    dephasing_rate = 0.1
    Jump_list = [np.sqrt(dephasing_rate) * np.array([[1, 0], [0, -1]])]


    # To do the tomography, we evolve each pauli operator
    TransferMatrix = np.zeros((4,4),dtype=complex)

    for i,sigma in enumerate(sigmas) :
        end_thingy = solver(sigma, lambda t, f: evolution_to_feed_rk4(t, f, lambda rho: master_derivative(rho, Hamiltonian, Jump_list), sigma.shape), t_span=[0, time])
        evolved_operator = ((end_thingy.y.T)[-1]).reshape(sigma.shape)
        column = state_tomography(evolved_operator)
        TransferMatrix[i] = column
    for i in range(0,4) :
        for j in range(0,4) :
            TransferMatrix[i,j] = TransferMatrix[i,j]/2
    print('\n Our pretty transfer matrix is \n')
    printArray(TransferMatrix)