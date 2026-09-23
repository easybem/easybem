// Dalibor Lukas, September 2026


#include <iostream>

#include "func.h"
#include "meshTopology.h"


// Mesh topology, common-edge and common-node pairs of elements
int nnodes, nedges, ne;
double *nodes;
int *elemNodes;
int *edges, *elemEdges;
int *nodeElems, *nodeElemsSizes, *nodeElemsIdx;
int *edgeElems, *commEdgeElems2Elems;
int *commNodeElems2Elems, *commNodeElems2ElemsSizes, *commNodeElems2ElemsIdx;
int commNodesNTotal;
long int disjPanelsNTotal;


void CreateEdges ()
{
  elemEdges = new int[3*ne];
  int *tmpEdges;
  tmpEdges = new int[2*3*ne];
  nedges = 0;
  for (int i=0; i<ne; i++)
    {
      int nodesi[3] = {elemNodes[3*i],elemNodes[3*i+1],elemNodes[3*i+2]};
      for (int j=0; j<3; j++)
        {
          int n1 = nodesi[j];
          int n2 = nodesi[(j+1)%3];
          int edge = 0;
          for (int k=0; k<nedges; k++)
            {
              if (tmpEdges[2*k] == n1 && tmpEdges[2*k+1] == n2)
                {
                  edge = k+1; break;
                }
              if (tmpEdges[2*k] == n2 && tmpEdges[2*k+1] == n1)
                {
                  edge = -(k+1); break;
                }
            }
          if (!edge)
            {
              tmpEdges[2*nedges] = n1;
              tmpEdges[2*nedges+1] = n2;
              edge = ++nedges;
            }
          elemEdges[3*i+j] = edge;
        }
    }
  edges = new int[2*nedges];
  memcpy(edges,tmpEdges,2*nedges*sizeof(int));
  delete [] tmpEdges;
}


void CreateEdgeElems ()
{
  int i, j, ej, sej;
  edgeElems = new int[2*nedges];
  memset(edgeElems,0,2*nedges*sizeof(int));
  for (i=0; i<ne; i++)
    for (j=0; j<3; j++)
      {
        ej = elemEdges[3*i+j];
        sej = sign(ej);
        ej = abs(ej);
        if (sej==1)
          edgeElems[2*(ej-1)] = i+1;
        else
          edgeElems[2*(ej-1)+1] = i+1;
      }
}


void CreateCommEdgeElems2Elems ()
{
  int i, idx1, idx2, j;
  commEdgeElems2Elems = new int[3*ne];
  memset(commEdgeElems2Elems,0,3*ne*sizeof(int));
  for (i=0; i<nedges; i++)
    {
      idx1 = edgeElems[2*i];
      idx2 = edgeElems[2*i+1];
      for (j=0; commEdgeElems2Elems[3*(idx1-1)+j]; j++) ;
      commEdgeElems2Elems[3*(idx1-1)+j] = idx2;
      for (j=0; commEdgeElems2Elems[3*(idx2-1)+j]; j++) ;
      commEdgeElems2Elems[3*(idx2-1)+j] = idx1;
    }
}


void CreateNodeElems ()
{
  int i, j, nj, idx;
  int nnod = 0;
  int tmp[nnodes];
  for (i=0; i<ne; i++)
    for (j=0; j<3; j++)
      Insert(tmp,&nnod,elemNodes[3*i+j]);
  nodeElemsSizes = new int[nnod];
  memset(nodeElemsSizes,0,nnod*sizeof(int));
  for (i=0; i<ne; i++)
    for (j=0; j<3; j++)
      {
        nj = elemNodes[3*i+j];
        nj = Find(nj,tmp,nnod);
        nodeElemsSizes[nj-1]++;
      }
  nodeElemsIdx = new int[nnod];
  for (i=0, idx=0; i<nnod; i++)
    {
      nodeElemsIdx[i] = idx;
      idx += nodeElemsSizes[i];
    }
  memset(nodeElemsSizes,0,nnod*sizeof(int));
  nodeElems = new int[3*ne];
  for (i=0; i<ne; i++)
    for (j=0; j<3; j++)
      {
        nj = elemNodes[3*i+j];
        nj = Find(nj,tmp,nnod);
        idx = nodeElemsIdx[nj-1] + nodeElemsSizes[nj-1];
        nodeElems[idx] = i+1;
        nodeElemsSizes[nj-1]++;
      }
}


void CreateCommNodeElems2Elems ()
{
  int i, idx, ni, j, fj, k, fk, ntotal, idxtotal;
  commNodeElems2ElemsSizes = new int[ne];
  memset(commNodeElems2ElemsSizes,0,ne*sizeof(int));
  for (i=0, ntotal=0; i<nnodes; i++)
    {
      ni = nodeElemsSizes[i];
      idx = nodeElemsIdx[i];
      for (j=0; j<ni; j++)
        {
          fj = nodeElems[idx+j];
          for (k=j+1; k<ni; k++)
            {
              fk = nodeElems[idx+k];
              if (commEdgeElems2Elems[3*(fj-1)]!=fk &&
                  commEdgeElems2Elems[3*(fj-1)+1]!=fk &&
                  commEdgeElems2Elems[3*(fj-1)+2]!=fk)
                {
                  commNodeElems2ElemsSizes[fj-1]++;
                  commNodeElems2ElemsSizes[fk-1]++;
                  ntotal += 2;
                }		
            }
        }
    }
  commNodeElems2ElemsIdx = new int[ne];
  for (i=0, idx=0; i<ne; i++)
    {
      commNodeElems2ElemsIdx[i] = idx;
      idx += commNodeElems2ElemsSizes[i];
    }
  commNodeElems2Elems = new int[ntotal];
  memset(commNodeElems2ElemsSizes,0,ne*sizeof(int));
  for (i=0; i<nnodes; i++)
    {
      ni = nodeElemsSizes[i];
      idx = nodeElemsIdx[i];
      for (j=0; j<ni; j++)
        {
          fj = nodeElems[idx+j];
          for (k=j+1; k<ni; k++)
            {
              fk = nodeElems[idx+k];
              if (commEdgeElems2Elems[3*(fj-1)]!=fk &&
                  commEdgeElems2Elems[3*(fj-1)+1]!=fk &&
                  commEdgeElems2Elems[3*(fj-1)+2]!=fk)
                {
                  idxtotal = commNodeElems2ElemsIdx[fj-1];
                  InsertSorted<int>(fk,commNodeElems2Elems+idxtotal,
                                    commNodeElems2ElemsSizes[fj-1]);
                  idxtotal = commNodeElems2ElemsIdx[fk-1];
                  InsertSorted<int>(fj,commNodeElems2Elems+idxtotal,
                                    commNodeElems2ElemsSizes[fk-1]);
                }
            }
        }
    }
}


void createMeshTopology (int annodes, const double *anodes,
                         int ane, const int *anelemNodes)
{
  int i, j, idx;
  nnodes = annodes;
  nodes = new double[3*nnodes];
  memcpy(nodes,anodes,3*nnodes*sizeof(double));
  ne = ane;
  elemNodes = new int[3*ne];
  memcpy(elemNodes,anelemNodes,3*ne*sizeof(int));
  /*
  std::cout << "Nodes: " << nnodes << std::endl;
  for (i=0; i<nnodes; i++)
    std::cout << nodes[3*i] << "," << nodes[3*i+1] << "," << nodes[3*i+2] << std::endl;
  std::cout << "Elements: " << ne << std::endl;
  for (i=0; i<ne; i++)
    std::cout << i+1 << ": " << elemNodes[3*i] << "," << elemNodes[3*i+1] << "," << elemNodes[3*i+2] << std::endl;
  */
  CreateEdges();
  /*
  std::cout << "Edges: " << nedges << std::endl;
  for (i=0; i<nedges; i++)
    std::cout << i+1 << ": " << edges[2*i] << "," << edges[2*i+1] << std::endl;
  for (i=0; i<ne; i++)
    std::cout << i+1 << ": " << elemEdges[3*i] << "," << elemEdges[3*i+1] << "," << elemEdges[3*i+2] << std::endl;
  */
  CreateEdgeElems();
  /*
  std::cout << "Edge-elements: " << nedges << std::endl;
  for (i=0; i<nedges; i++)
    std::cout << i+1 << ": " << edgeElems[2*i] << "," << edgeElems[2*i+1] << std::endl;
  */
  CreateCommEdgeElems2Elems();
  /*
  std::cout << "Common-edge elems-to-elems: " << ne << std::endl;
  for (i=0; i<ne; i++)
    std::cout << i+1 << ": " << commEdgeElems2Elems[3*i] << "," << commEdgeElems2Elems[3*i+1] << "," << commEdgeElems2Elems[3*i+2] << std::endl;
  */
  CreateNodeElems();
  /*
  std::cout << "Nodes-to-elements: " << nnodes << std::endl;
  for (i=0; i<nnodes; i++)
    {
      std::cout << i+1 << ": ";
      int idx = nodeElemsIdx[i];
      for (int j=0; j<nodeElemsSizes[i]; j++)
        std::cout << nodeElems[idx+j] << ",";
      std::cout << std::endl;
    }
  */
  CreateCommNodeElems2Elems();
  /*
  std::cout << "Common-node elems-to-elems: " << ne << std::endl;
  for (i=0; i<ne; i++)
    {
      std::cout << i+1 << ": ";
      int idx = commNodeElems2ElemsIdx[i];
      for (int j=0; j<commNodeElems2ElemsSizes[i]; j++)
        std::cout << commNodeElems2Elems[idx+j] << ",";
      std::cout << std::endl;
    }
  */
  commNodesNTotal = commNodeElems2ElemsIdx[ne-1] +
    commNodeElems2ElemsSizes[ne-1];
  disjPanelsNTotal = ((long int) ne)*(((long int)ne)-4)-commNodesNTotal;
}


void releaseMeshTopology ()
{
  delete [] nodes;
  delete [] elemNodes;
  delete [] edges;
  delete [] elemEdges;
  delete [] edgeElems;
  delete [] commEdgeElems2Elems;
  delete [] nodeElems;
  delete [] nodeElemsSizes;
  delete [] nodeElemsIdx;
  delete [] commNodeElems2Elems;
  delete [] commNodeElems2ElemsSizes;
  delete [] commNodeElems2ElemsIdx;
}
