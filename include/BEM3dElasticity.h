// Dalibor Lukas, May 2026

#ifndef BEM3DELASTICITY_H
#define BEM3DELASTICITY_H


void getElasticity3d (int nNodes, const double *nodes,
                      int nElements, const int *elements,
                      double nu, // E = 1.0
                      double *&V, double *&K, double *&D, double *&M,
                      int order);

/*
void evaluateElasticity3d (const double *nodes,
                           int nElements, const int *elements,
                           int nPoints, const double *points,
                           double E, double nu,
                           const double *neumann, const double *dirichlet,
                           double *&result,
                           int order);
*/

#endif
