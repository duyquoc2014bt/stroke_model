import numpy as np
import utils  

A = np.array([1, 2, 3])
B = np.array([4, 5, 6])

def calc_cosin(A, B):
    dot_A_B = np.dot(utils.unit_vector(A), utils.unit_vector(B))
    return dot_A_B

print(calc_cosin(A, B))