import numpy as np
from scipy.sparse.linalg import LinearOperator, cg
import surface3dMesh as sm
import time

import sys
sys.path.insert(0, "../../build/python")
import easybem as eb

nref = 1
order = 4

#file = "cube01-surface.txt"
##file = "cube01_2ref-surface.txt"
#P,T,Tidx = sm.load3dSurface(file)

file = "shaft-surface_verycoarse.vol"
P,T,Tidx = sm.load3dSurfaceFromNetgen(file)

for i in range(nref):
    P,T,Tidx = sm.refineMesh(P,T,Tidx)

    
tic = time.perf_counter()
V,K,D,M = eb.get_laplace3d(P,T,order)
toc = time.perf_counter()
print(f"{(toc-tic):.6f} seconds")

nnodes = P.shape[0]
ne = T.shape[0]
print("nnodes=",nnodes,"; ne=",ne)

A = P[T[:, 0]-1]
B = P[T[:, 1]-1]
C = P[T[:, 2]-1]
areas = 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=1)
m = np.zeros(nnodes)
np.add.at(m, T-1, areas[:, None] / 3)
C1,C2,C3 = eb.get_laplace3d_curl(P,T)

eb.init_gpu(P,T,order,240,32)

def applyOperator(x):
    return (
        C1.T @ eb.apply_gpu_laplace3d_1layer(C1 @ x)
        + C2.T @ eb.apply_gpu_laplace3d_1layer(C2 @ x)
        + C3.T @ eb.apply_gpu_laplace3d_1layer(C3 @ x)
        + m* (m @ x)
    )


def applyPreconditioner(x):
    return eb.apply_gpu_laplace3d_1layer_linears(x)


D = LinearOperator(
    shape=(nnodes, nnodes),
    matvec=applyOperator,
    dtype=np.float64
)

Prec = LinearOperator(
    shape=(nnodes, nnodes),
    matvec=applyPreconditioner,
    dtype=np.float64
)


b = np.ones(nnodes)


# -------------------------------------------------------
# CG
# -------------------------------------------------------
x0 = np.zeros(nnodes)

tic = time.perf_counter()


iteration_count = 0
def callback(xk):
    global iteration_count
    iteration_count += 1
x, info = cg(
    D,
    b,
    x0=np.zeros(nnodes),
    tol=1e-4,
    callback=callback
)

#x, info = cg(
#    D,
#    b,
#    x0=x0,
#    M=Prec,
#    tol=1e-8,
#    atol=0.0,
#    callback=callback
#)


toc = time.perf_counter()
print(f"{(toc-tic):.6f} seconds")

print("iterations =",iteration_count+1)
print("residual =", np.linalg.norm(b - applyOperator(x)))


eb.close_gpu()
