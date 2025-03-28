import numpy as np

La = np.array([0, 1/3, 1/3, 1/3])
Lb = np.array([1/2, 0, 0, 1/2])
Lc = np.array([0, 0, 0, 1])
Ld = np.array([0, 1/2, 1/2, 0])

L = np.column_stack((La, Lb, Lc, Ld))
rank = np.array([1/4, 1/4, 1/4, 1/4])

def page_rank(initial_rank, L, d):
    n = L.shape[1]
    new_rank = initial_rank
    for i in range(0, 99, 1):
        new_rank = d*(L@new_rank) + (1-d)/n
    return new_rank

print(page_rank(rank, L, 0.85))
