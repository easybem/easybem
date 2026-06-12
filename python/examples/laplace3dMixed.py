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
    -np.sum((P - xstar)*N,axis=1) /
    np.linalg.norm(P-xstar,axis=1)**3
)

nrefs = 5
order = 4
file = "cube01-surface.txt"

for i in range(nrefs):
    if i==0:
        P,T,Tidx = sm.load3dSurface(file)
    else:
        P,T,Tidx = sm.refineMesh(P,T,Tidx)

    npts = P.shape[0]
    nelems = T.shape[0]
    GammaDelems = np.where(Tidx==0)[0]
    nGammaDelems = GammaDelems.shape[0]
    GammaDpts = np.unique(T[Tidx==0])-1
    GammaNelems = np.where(Tidx!=0)[0]
    GammaNpts = np.unique(T[Tidx!=0])-1
    GammaNpts = np.setdiff1d(GammaNpts,GammaDpts)
    nGammaNpts = GammaNpts.shape[0]

    Tri = T-1
    PTri = P[Tri]
    Pcenters = PTri.mean(axis=1)
    J = np.cross(PTri[:,1]-PTri[:,0],PTri[:,2]-PTri[:,0])
    Areas = 0.5 * np.linalg.norm(J,axis=1)
    Normals = J / np.linalg.norm(J, axis=1, keepdims=True)

    V,K,D,M = eb.get_laplace3d(P,T,order+i)
    VDD = V[np.ix_(GammaDelems,GammaDelems)]
    KDN = K[np.ix_(GammaDelems,GammaNpts)]
    DNN = D[np.ix_(GammaNpts,GammaNpts)]
    Matrix = np.block([[VDD,-KDN],
                       [np.transpose(KDN),DNN]])
    
    uD = uexact(P[GammaDpts,:])
    uD = np.asarray(uD).reshape(-1)
    PN = Pcenters[GammaNelems,:]
    tN = texact(Pcenters[GammaNelems,:],Normals[GammaNelems,:])
    bD = (
        (0.5*M[np.ix_(GammaDelems,GammaDpts)] +
         K[np.ix_(GammaDelems,GammaDpts)]) @ uD
        - V[np.ix_(GammaDelems,GammaNelems)] @ tN
    )
    bN = (
        np.transpose(0.5*M[np.ix_(GammaNelems,GammaNpts)] -
                     K[np.ix_(GammaNelems,GammaNpts)]) @ tN -
        D[np.ix_(GammaNpts,GammaDpts)] @ uD
    )
    Vector = np.concatenate([
        np.asarray(bD).reshape(-1, 1),
        np.asarray(bN).reshape(-1, 1)
    ], axis=0)
    x = np.linalg.solve(Matrix,Vector)
    tD = x[:nGammaDelems]
    uN = x[nGammaDelems:]

    u = np.zeros(npts)
    u[GammaDpts] = uD.flatten()
    u[GammaNpts] = uN.flatten()
    t = np.zeros(nelems)
    t[GammaDelems] = tD.flatten()
    t[GammaNelems] = tN.flatten()

    tex = texact(Pcenters,Normals)
    L2error = np.sqrt(np.sum((t - tex)**2 * Areas))
    print("L2error ||t-texact||=",L2error)

    points = np.array([[0.5,0.5,0.5]])
    u0 = eb.eval_laplace3d(P,T,points,t,u,order+i)
    print(u0,uexact(points))

sm.plot3dSurface(P,T,t)

