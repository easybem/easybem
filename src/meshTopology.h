// Dalibor Lukas, September 2026


#ifndef MESHTOPOLOGY_H
#define MESHTOPOLOGY_H


extern int nnodes, nedges, ne;
extern double *nodes;
extern int *elemNodes;
extern int *edges, *elemEdges;
extern int *nodeElems, *nodeElemsSizes, *nodeElemsIdx;
extern int *edgeElems, *commEdgeElems2Elems;
extern int *commNodeElems2Elems, *commNodeElems2ElemsSizes;
extern int *commNodeElems2ElemsIdx;
extern int commNodesNTotal;
extern long int disjPanelsNTotal;


void createMeshTopology (int annodes, const double *anodes,
                         int ane, const int *anelemNodes);

void releaseMeshTopology();


#endif
