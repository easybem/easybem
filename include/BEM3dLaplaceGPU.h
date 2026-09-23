// Dalibor Lukas, September 2026


#ifndef BEM3DLAPLACEGPU_H
#define BEM3DLAPLACEGPU_H


void initGPU (int nNodes, const double *nodes,
              int nElements, const int *elements, int order,
              int nblocks, int nthreads);

void closeGPU ();

void applyGPULaplace3d1Layer (const double *t, double *v, int ansatz=0);

void applyGPULaplace3d2Layer (const double *u, double *q);

void applyGPULaplace3dHyperSing (const double *u, double *q);


#endif
