import math
import numpy as np
import surface3dMesh as sm
from scipy.sparse.linalg import cg, LinearOperator

import sys
sys.path.insert(0, "../../build/python")
import easybem as eb

#uexact = lambda P: P[:,0]+P[:,1]+P[:,2]
#texact = lambda P,N: N[:,0]+N[:,1]+N[:,2]

xstar = np.array([2.0, 2.0, 2.0])
uexact = lambda P: 1.0 / np.linalg.norm(P - xstar, axis=1)
texact = lambda P,N: (
    (-np.sum((P - xstar)*N,axis=1) /
     np.linalg.norm(P-xstar,axis=1)**3)
    )

nrefs = 5
order = 8
file = "cube01-surface.txt"
#file = "shaft-surface_verycoarse.vol"

for i in range(nrefs):
    if i==0:
        P,T,Tidx = sm.load3dSurface(file)
        #P,T,Tidx = sm.load3dSurfaceFromNetgen(file)
        #P = P/100
    else:
        P,T,Tidx = sm.refineMesh(P,T,Tidx)

    nnodes = P.shape[0]
    ne = T.shape[0]
    _,K,D,M = eb.get_laplace3d(P,T,order+i)
    M1 = sm.surface_mass_matrix(P,T)
    
    u = uexact(P)
    Tri = T-1
    PTri = P[Tri]
    Pcenters = PTri.mean(axis=1)
    J = np.cross(PTri[:,1]-PTri[:,0],PTri[:,2]-PTri[:,0])
    Areas = 0.5 * np.linalg.norm(J,axis=1)
    Normals = J / np.linalg.norm(J, axis=1, keepdims=True)
    t = texact(Pcenters,Normals)
    m = np.zeros(nnodes)
    np.add.at(m, Tri, Areas[:, None] / 3)

    A = D + np.outer(m,m)
    b = (0.5*M.T-K.T)@t + m
    
    uh = np.linalg.solve(A,b)
    uh_zero_mean = uh - (m @ uh) / m.sum()
    u_zero_mean = u - (m @ u) / m.sum()
    err = uh_zero_mean - u_zero_mean
    L2error = np.sqrt(err@(M1@err))
    print("L2error ||u-uexact||/||uexact||=",
          L2error/np.sqrt(u_zero_mean@(M1@u_zero_mean)))

    iteration_count = 0
    def callback(xk):
        global iteration_count
        iteration_count += 1
    u, info = cg(
        A,
        b,
        x0=np.zeros(nnodes),
        tol=1e-8,
        callback=callback
    )
    print("iterations =",iteration_count+1)
    print("residual =", np.linalg.norm(b - A@u))

    
    input("Press a key to continue")

sm.plot3dSurface(P,T,t)
