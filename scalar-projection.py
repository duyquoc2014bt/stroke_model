import numpy as np
import utils

A = np.array([3, 4])
B = np.array([1, 2])

def scalar_projection(A, B, isVector = False, *kwargs):
    unit_vectorB = utils.unit_vector(B)
    dot_A_unit_vectorB = np.dot(A, unit_vectorB)
    
    if isVector:
        return dot_A_unit_vectorB*unit_vectorB
    return dot_A_unit_vectorB

print(scalar_projection(A, B), scalar_projection(A, B, True))

