import numpy as np

A = np.array([[4, 7], 
              [2, 6]], dtype=np.float64)

def inverse_matrix(A): 
    if A.shape[0] != A.shape[1]:
        raise ValueError("The matrix must be a square matrix!")
    
    detA = np.linalg.det(A)
    if detA == 0:
        raise ValueError("The vectors in the matrix are not linearly independent vectors!")
    
    return np.linalg.inv(A)

print(inverse_matrix(A))