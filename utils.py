import numpy as np

def unit_vector(V): 
    norm_V = np.linalg.norm(V)
    if norm_V == 0:
        raise ValueError("Vector of magnitude zero cannot be normalized!")
    return V/norm_V

def calc_cosin(A, B):
    dot_A_B = np.dot(unit_vector(A), unit_vector(B))
    return dot_A_B


def scalar_projection(A, B, isVector = False):
    unit_vectorB = unit_vector(B)
    dot_A_unit_vectorB = np.dot(A, unit_vectorB)
    
    if isVector:
        return dot_A_unit_vectorB*unit_vectorB
    return dot_A_unit_vectorB

def re_changing_basis(re, b1, b2):
    re_b1_projection = scalar_projection(re, b1)/np.linalg.norm(b1)
    re_b2_projection = scalar_projection(re, b2)/np.linalg.norm(b2)
    return np.array([re_b1_projection, re_b2_projection])

# A-1 khong co y nghia tinh toan, nen inv(A) = AT = A-1/|A|
def inverse_matrix(A): 
    if A.shape[0] != A.shape[1]:
        raise ValueError("The matrix must be a square matrix!")
    
    detA = np.linalg.det(A)
    if detA == 0:
        raise ValueError("The vectors in the matrix are not linearly independent vectors!")
    
    return np.linalg.inv(A)

def gram_schmidt(vectors): #gram-schmidt
    m, n = vectors.shape
    basis = np.zeros((m, n))
    
    for i in range(n):
        w = vectors[:, i]
        
        for j in range(i):
            try:
                w = w - scalar_projection(vectors[:, i], basis[:, j], True)
            except ValueError as e:
                print(e)
            
            
        w = unit_vector(w)
        basis[:, i] = w
        
    return np.array(basis)

def gram_schmidt_secondary(vectors): #householder/given
    return  np.linalg.qr(vectors)

def qr_decomposition(A):
    Q = gram_schmidt(A)
    R = np.dot(Q.T, A)
    return np.array([Q, R])

def page_rank(initial_rank, L, d):
    n = L.shape[1]
    new_rank = initial_rank
    for i in range(0, 99, 1):
        new_rank = d*(L@new_rank) + (1-d)/n
    return new_rank

