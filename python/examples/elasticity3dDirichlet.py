import numpy as np
import surface3dMesh as sm

import sys
sys.path.insert(0, "../../build/python")
import easybem as eb

nu = 0.3
uexact = lambda P: 1.0 * (P * np.linalg.norm(P-np.array([0.5,0.5,0.5]),axis=1)
                          [:,None])


nrefs = 4
order = 7
file = "cube01-surface.txt"
#file = "shaft-surface.txt"

P,T,Tidx = sm.load3dSurface(file)
for i in range(nrefs):
    P,T,Tidx = sm.refineMesh(P,T,Tidx)
n = P.shape[0]
m = T.shape[0]

Tri = T-1
PTri = P[Tri]
Pcenters = PTri.mean(axis=1)
J = np.cross(PTri[:,1]-PTri[:,0],PTri[:,2]-PTri[:,0])
Areas = 0.5 * np.linalg.norm(J,axis=1)
Normals = J / np.linalg.norm(J, axis=1, keepdims=True)

V,K,_,M = eb.get_elasticity3d(P,T,nu,order)
U = uexact(P)
u = U.flatten()
t = np.linalg.solve(V,(0.5*M+K)@u)

sm.plot3dSurface(P+U,T, np.sum(t.reshape(m,3)*Normals,axis=1))
