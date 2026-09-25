import math
import numpy as np
from scipy.sparse.linalg import cg, LinearOperator
import surface3dMesh as sm
import time

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
#file = "cube01-surface.txt"
file = "shaft-surface_verycoarse.vol"
order = 9

for i in range(nrefs):
    if i==0:
        #P,T,Tidx = sm.load3dSurface(file)
        P,T,Tidx = sm.load3dSurfaceFromNetgen(file)
        P = P/100
    else:
        P,T,Tidx = sm.refineMesh(P,T,Tidx)

    nnodes = P.shape[0]
    ne = T.shape[0]
    u = uexact(P)
    Tri = T-1
    PTri = P[Tri]
    Pcenters = PTri.mean(axis=1)
    J = np.cross(PTri[:,1]-PTri[:,0],PTri[:,2]-PTri[:,0])
    Areas = 0.5 * np.linalg.norm(J,axis=1)
    Normals = J / np.linalg.norm(J, axis=1, keepdims=True)
    tex = texact(Pcenters,Normals)
    sm.plot3dSurface(P,T)
    
    if i==0:
        tic = time.perf_counter()
        V,K,_,M = eb.get_laplace3d(P,T,order+i)
        b = (0.5*M+K)@u
        t = np.linalg.solve(V,b)
        toc = time.perf_counter()
        print(f"CPU-BEM solution in {(toc-tic):.6f} seconds")
        L2error = np.sqrt(np.sum((t-tex)**2 * Areas))
        L2error /= np.sqrt(np.sum((tex)**2 * Areas))
        print("L2error ||t-texact||=",L2error)

        input("Press a key...")

    MM = eb.get_laplace3d_mass(P,T)
    tic = time.perf_counter()
    eb.init_gpu(P,T,order+i,240,32)
    b = (0.5 * (MM@u)) +  eb.apply_gpu_laplace3d_2layer(u)    
    def applyOperator(x):
        return (eb.apply_gpu_laplace3d_1layer(x))
    VV = LinearOperator(
        shape=(ne, ne),
        matvec=applyOperator,
        dtype=np.float64
    )  
    iteration_count = 0
    def callback(xk):
        global iteration_count
        iteration_count += 1    
    t, info = cg(
        VV,
        b,
        x0=np.zeros(ne),
        tol=1e-4,
        callback=callback
    )
    print("iterations =",iteration_count+1)
    eb.close_gpu()
    toc = time.perf_counter()
    print(f"GPU-BEM solution in {(toc-tic):.6f} seconds")
    L2error = np.sqrt(np.sum((t-tex)**2 * Areas))
    L2error /= np.sqrt(np.sum((tex)**2 * Areas))
    print("L2error ||t-texact||=",L2error)

    input("Press a key...")
