import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt


'''
Here, the idea will be to use an RK45 solver
to solve the master equation for a given initial
state, Hamiltonian and list of jump operators.

Recipe is as follows:

1. Start from a density matrix rho_0
2. Vectorize it to feed it into the RK45 solver
3. The f(t, y) function will do the following:
    - Reshape the vector into a matrix
    - Compute the derivative of the density matrix
    - Reshape the matrix into a vector
4. Solve the ODE using the RK45 solver
'''

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
                      - 0.5 * np.dot(np.dot(jump_operator.conj().T, jump_operator), rho)
                      - 0.5 * np.dot(np.dot(rho, jump_operator.conj().T), jump_operator))

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

if __name__ == "__main__":

    # Initial state
    r_0 = np.array([1, 1, 5])
    r_0 = r_0 / np.linalg.norm(r_0)
    rho_0 = bloch_vector_to_density(r_0)

    print ('\nInitial state:\n', rho_0)

    # Evolution, please define also the correct jump rates
    time = 10
    Hamiltonian = np.array([[1, 0], [0, -1]])  # Evolution is a simple rotation around the z-axis
    dephasing_rate = 0.1
    Jump_list = [np.sqrt(dephasing_rate) * np.array([[1, 0], [0, -1]])]

    print('\nHamiltonian:\n', Hamiltonian)
    
    # Derivative test
    print('\nDerivative test:\n', master_derivative(rho_0, Hamiltonian, Jump_list))

    # Small test
    vect_rho_0 = rho_0.reshape(rho_0.size)
    print('\nVectorized rho_0:\n', vect_rho_0)
    devect_rho_0 = vect_rho_0.reshape(rho_0.shape)
    print('\nDevectorized rho_0:\n', devect_rho_0)
    print('\nAre they equal?', np.allclose(rho_0, devect_rho_0))
          
    # Solve the master equation
    solution = solver(rho_0, lambda t, f: evolution_to_feed_rk4(t, f, lambda rho: master_derivative(rho, Hamiltonian, Jump_list), rho_0.shape), t_span=[0, time])

    # Now, this solution is a list of vectors, we need to reshape them into matrices !
    density_matrices = np.array([rho.reshape(rho_0.shape) for rho in solution.y.T])

    # Compute the Bloch vector components
    bloch_x = np.array([np.trace(np.dot(density, pauli_x)).real for density in density_matrices])
    bloch_y = np.array([np.trace(np.dot(density, pauli_y)).real for density in density_matrices])
    bloch_z = np.array([np.trace(np.dot(density, pauli_z)).real for density in density_matrices])

    # Plot the evolution of each component of the Bloch vector
    plt.plot(solution.t, bloch_x, label=r"$\langle \sigma_x \rangle$")
    plt.plot(solution.t, bloch_y, label=r"$\langle \sigma_y \rangle$")
    plt.plot(solution.t, bloch_z, label=r"$\langle \sigma_z \rangle$")

    plt.xlabel("Time")
    plt.ylabel("Bloch vector components")
    plt.legend()
    plt.title("Evolution of Bloch Vector Components")
    plt.grid()
    plt.savefig("bloch_vector_evolution.png")
    plt.close()
