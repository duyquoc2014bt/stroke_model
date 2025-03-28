import numpy as np
import utils

re = np.array([3, 4])
b1 = np.array([2, 1])
b2 = np.array([-2, 4])

#first way
def re_changing_basis(re, b1, b2):
    re_b1_projection = utils.scalar_projection(re, b1)/np.linalg.norm(b1)
    re_b2_projection = utils.scalar_projection(re, b2)/np.linalg.norm(b2)
    return np.array([re_b1_projection, re_b2_projection])

print(re_changing_basis(re, b1, b2))