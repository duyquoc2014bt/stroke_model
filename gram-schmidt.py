import numpy as np
import utils

A = np.array([
    [3, 1, 1],
    [2, 2, 1],
    [1, 1, 3]
])

def gram_schmidt(vectors): #gram-schmidt
    m, n = A.shape
    basis = np.zeros((m, n))
    
    for i in range(n):
        w = vectors[:, i]
        
        for j in range(i):
            try:
                w = w - utils.scalar_projection(vectors[:, i], basis[:, j], True)
            except ValueError as e:
                print(e)
            
            
        w = utils.unit_vector(w)
        basis[:, i] = w
        
    return np.array(basis)

def gram_schmidt_secondary(vectors): #householder/given
    return  np.linalg.qr(vectors)

def qr_decomposition(A):
    Q = gram_schmidt(A)
    R = np.dot(Q.T, A)
    return np.array([Q, R])


# print(gram_schmidt(A))
print(gram_schmidt_secondary(A)[0])
print(qr_decomposition(A)[0])
