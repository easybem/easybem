import math
import numpy as np
import surface3dMesh as sm

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

    sm.plot3dSurface(P,T)
    V,K,_,M = eb.get_laplace3d(P,T,order+i)
    u = uexact(P)
    t = np.linalg.solve(V,(0.5*M+K)@u)

    Tri = T-1
    PTri = P[Tri]
    Pcenters = PTri.mean(axis=1)
    J = np.cross(PTri[:,1]-PTri[:,0],PTri[:,2]-PTri[:,0])
    Areas = 0.5 * np.linalg.norm(J,axis=1)
    Normals = J / np.linalg.norm(J, axis=1, keepdims=True)
    tex = texact(Pcenters,Normals)
    L2error = np.sqrt(np.sum((t-tex)**2 * Areas))
    print("L2error ||t-texact||=",L2error)

    points = np.array([[0.5,0.5,0.5]])
    u0 = eb.eval_laplace3d(P,T,points,t,u,order+i)
    print(u0,uexact(points))

sm.plot3dSurface(P,T,t)
