import numpy as np
from scipy import sparse
from scipy.sparse import csc_matrix, linalg as sla
from scipy.linalg import lu
import time


sparse_matrix=np.zeros((10000,10000))
for i in range(10000):
    sparse_matrix[i % 10000][(i + 1) % 10000] = 1

b = csc_matrix(sparse_matrix)
splu_start = time.time()
res = sla.splu(b)
splu_end = time.time()
lu_start = time.time()
p,l,u = lu(sparse_matrix)
lu_end = time.time()

print(f"use time:{splu_end - splu_start}")
print(f"use time:{lu_end - lu_start}")