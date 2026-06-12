// Dalibor Lukas, May 2026


#ifndef BEM3DLAPLACE_H
#define BEM3DLAPLACE_H


void getLaplace3d (int nNodes, const double *nodes,
                   int nElements, const int *elements,
                   double *&V, double *&K, double *&D, double *&M,
                   int order);

void evaluateLaplace3d (const double *nodes,
                        int nElements, const int *elements,
                        int nPoints, const double *points,
                        const double *neumann, const double *dirichlet,
                        double *&result,
                        int order);

#endif
