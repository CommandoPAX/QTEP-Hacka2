import numpy as np
import scipy.linalg as la

def ket_to_density(ket):
    return np.outer(ket, ket.conj())



x = np.array([0,1])
print(ket_to_density(x))