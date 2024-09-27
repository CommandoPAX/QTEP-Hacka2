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

def Channel(rho : np.array, Hamilt : np.array) : 
    return np.dot(np.dot(Hamilt, rho), Hamilt.conj().T)

def R(Lambda : np.array) : 
    TransferMatrix = np.zeros((4,4),dtype=complex)
    sigmas = [identity, pauli_x, pauli_y, pauli_z]
    for i in range(0,4) : 
        for j in range(0,4) : 
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
    print("'\nTransfer Matrix:\n", R(Hamiltonian))