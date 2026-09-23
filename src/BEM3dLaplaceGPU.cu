// Dalibor Lukas, September 2026

//using namespace std;


#include <iostream>

#include "quadrature.h"
#include "meshTopology.h"
#include "BEM3dLaplaceGPU.h"


// CUDA batch size: nn=nblocks*nthreads
int nblocks, nthreads, nn;
  
// CUDA shared memory
__constant__ float quadPoints2[2];
__constant__ float quadWeights2[2];
__constant__ float quadPoints3[3];
__constant__ float quadWeights3[3];
__constant__ float quadPoints4[4];
__constant__ float quadWeights4[4];
__constant__ float quadPoints[64];
__constant__ float quadWeights[64];
__constant__ float quadTriPoints1[79];
__constant__ float quadTriPoints2[79];
__constant__ float quadTriWeights[79];

// CPU-from/to-GPU batch data
float *As, *us, *vs, *Bs, *ws, *zs, *Matrix;
float *dev_As, *dev_us, *dev_vs, *dev_Bs, *dev_ws, *dev_zs, *dev_Matrix;

// Gauss quadratures
float *quPoints2, *quWeights2;
float *quPoints3, *quWeights3;
float *quPoints4, *quWeights4;
extern int quadn;
float *quPoints, *quWeights;
extern int quadTrin;
float *quTriPoints1, *quTriPoints2, *quTriWeights;



#define F_PI ((float)(M_PI))


__global__ void singleLayerLaplace3d_idPanels(float *us, float *vs, int quadn,
                                              float *V)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int idx = 3*tid, i;
  float u1, u2, u3, v1, v2, v3, J1, J2, J3, Jsq;
  float xy1, xy2, xy3, I, Ii, eta3;
  
  u1 = us[idx]; u2 = us[idx+1]; u3 = us[idx+2];
  v1 = vs[idx]; v2 = vs[idx+1]; v3 = vs[idx+2];
  J1 = u2*v3-u3*v2; J2 = u3*v1-u1*v3; J3 = u1*v2-u2*v1;
  Jsq = J1*J1+J2*J2+J3*J3;

  for (i=0, I=0.0f; i<quadn; i++)
    {
      eta3 = quadPoints[i];
      xy1 = eta3*u1+v1;
      xy2 = eta3*u2+v2;
      xy3 = eta3*u3+v3;
      Ii = 1.0f/sqrtf(xy1*xy1+xy2*xy2+xy3*xy3);
      xy1 = eta3*v1+u1;
      xy2 = eta3*v2+u2;
      xy3 = eta3*v3+u3;
      Ii += 1.0f/sqrtf(xy1*xy1+xy2*xy2+xy3*xy3);
      xy1 = eta3*(u1+v1)-v1;
      xy2 = eta3*(u2+v2)-v2;
      xy3 = eta3*(u3+v3)-v3;
      Ii += 1.0f/sqrtf(xy1*xy1+xy2*xy2+xy3*xy3);
      I += quadWeights[i]*Ii;
    }

  V[tid] = Jsq * I / F_PI / 12.0f;
}


__global__ void singleLayer1Laplace3d_idPanels(float *us, float *vs, int quadn,
                                               float *V)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int idx = 3*tid, i;
  float u1, u2, u3, v1, v2, v3, J1, J2, J3, Jsq;
  float xy1, xy2, xy3, eta3, weta3;
  float I, I11, I12, I13, I21, I22, I23, I31, I32, I33;
  
  u1 = us[idx]; u2 = us[idx+1]; u3 = us[idx+2];
  v1 = vs[idx]; v2 = vs[idx+1]; v3 = vs[idx+2];
  J1 = u2*v3-u3*v2; J2 = u3*v1-u1*v3; J3 = u1*v2-u2*v1;
  Jsq = J1*J1+J2*J2+J3*J3;
  I11 = I12 = I13 = I21 = I22 = I23 = I31 = I32 = I33 = 0.0f;

  for (i=0; i<quadn; i++)
    {
      // D1
      eta3 = quadPoints[i];
      weta3 = quadWeights[i];
      xy1 = eta3*u1+v1;
      xy2 = eta3*u2+v2;
      xy3 = eta3*u3+v3;
      I = weta3 / sqrtf(xy1*xy1+xy2*xy2+xy3*xy3);
      I11 += (1.0f/30.0f) * I;
      I12 += (1.0f/60.0f) * I;
      I13 += (1.0f/30.0f) * I;
      I21 += (1.0f/60.0f) * I;
      I22 += (1.0f/30.0f) * I;
      I23 += (1.0f/30.0f) * I;
      I31 += (1.0f/30.0f) * I;
      I32 += (1.0f/30.0f) * I;
      I33 += (1.0f/10.0f) * I;
      // D2
      xy1 = eta3*v1+u1;
      xy2 = eta3*v2+u2;
      xy3 = eta3*v3+u3;
      I = weta3 / sqrtf(xy1*xy1+xy2*xy2+xy3*xy3);
      I11 += (1.0f/30.0f) * I;
      I12 += ((2.0f-eta3)/60.0f) * I;
      I13 += ((1.0f+eta3)/60.0f) * I;
      I21 += ((2.0f-eta3)/60.0f) * I;
      I22 += ((3.0f-3.0f*eta3+eta3*eta3)/30.0f) * I;
      I23 += ((1.0f+eta3-eta3*eta3)/30.0f) * I;
      I31 += ((1.0f+eta3)/60.0f) * I;
      I32 += ((1.0f+eta3-eta3*eta3)/30.0f) * I;
      I33 += ((1.0f+eta3+eta3*eta3)/30.0f) * I;
      // D3
      xy1 = eta3*(u1+v1)-v1;
      xy2 = eta3*(u2+v2)-v2;
      xy3 = eta3*(u3+v3)-v3;
      I = weta3 / sqrtf(xy1*xy1+xy2*xy2+xy3*xy3);
      I11 += ((1.0f+eta3+eta3*eta3)/30.0f) * I;
      I12 += ((1.0f+eta3)/60.0f) * I;
      I13 += ((1.0f+eta3-eta3*eta3)/30.0f) * I;
      I21 += ((1.0f+eta3)/60.0f) * I;
      I22 += (1.0f/30.0f) * I;
      I23 += ((2.0f-eta3)/60.0f) * I;
      I31 += ((1.0f+eta3-eta3*eta3)/30.0f) * I;
      I32 += ((2.0f-eta3)/60.0f) * I;
      I33 += ((3.0f-3.0f*eta3+eta3*eta3)/30.0f) * I;
    }

  V[9*tid] = Jsq * I11 / F_PI / 4.0f;
  V[9*tid+1] = Jsq * I21 / F_PI / 4.0f;
  V[9*tid+2] = Jsq * I31 / F_PI / 4.0f;
  V[9*tid+3] = Jsq * I12 / F_PI / 4.0f;
  V[9*tid+4] = Jsq * I22 / F_PI / 4.0f;
  V[9*tid+5] = Jsq * I32 / F_PI / 4.0f;
  V[9*tid+6] = Jsq * I13 / F_PI / 4.0f;
  V[9*tid+7] = Jsq * I23 / F_PI / 4.0f;
  V[9*tid+8] = Jsq * I33 / F_PI / 4.0f;
}


__global__ void singleLayerLaplace3d_commonEdge(float *us, float *vs,
                                                float *ws, int quadn,
                                                float *V)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int idx = 3*tid, i, j;
  float u1, u2, u3, v1, v2, v3, w1, w2, w3, J1, J2, J3, J;
  float xy1, xy2, xy3, I, Iij, eta2, eta3, weta2, weta3;
  
  u1 = us[idx]; u2 = us[idx+1]; u3 = us[idx+2];
  v1 = vs[idx]; v2 = vs[idx+1]; v3 = vs[idx+2];
  w1 = ws[idx]; w2 = ws[idx+1]; w3 = ws[idx+2];
  J1 = u2*v3-u3*v2; J2 = u3*v1-u1*v3; J3 = u1*v2-u2*v1;
  J = sqrtf(J1*J1+J2*J2+J3*J3);
  J1 = u2*w3-u3*w2; J2 = u3*w1-u1*w3; J3 = u1*w2-u2*w1;
  J *= sqrtf(J1*J1+J2*J2+J3*J3);

  for (i=0, I=0.0f; i<quadn; i++)
    {
      eta2 = quadPoints[i];
      weta2 = quadWeights[i];
      for (j=0; j<quadn; j++)
        {
          eta3 = quadPoints[j];
          weta3 = quadWeights[j];
          xy1 = eta2*(eta3*(u1+w1)-w1)+v1;
          xy2 = eta2*(eta3*(u2+w2)-w2)+v2;
          xy3 = eta2*(eta3*(u3+w3)-w3)+v3;
          Iij = 1.0f/sqrtf(xy1*xy1+xy2*xy2+xy3*xy3);
          xy1 = -eta2*(u1+v1+eta3*w1)+v1;
          xy2 = -eta2*(u2+v2+eta3*w2)+v2;
          xy3 = -eta2*(u3+v3+eta3*w3)+v3;
          Iij += 1.0f/sqrtf(xy1*xy1+xy2*xy2+xy3*xy3);
          xy1 = eta2*(v1-eta3*(u1+v1))-w1;
          xy2 = eta2*(v2-eta3*(u2+v2))-w2;
          xy3 = eta2*(v3-eta3*(u3+v3))-w3;
          Iij += 1.0f/sqrtf(xy1*xy1+xy2*xy2+xy3*xy3);
          xy1 = -eta2*(w1+eta3*(u1+v1))+v1;
          xy2 = -eta2*(w2+eta3*(u2+v2))+v2;
          xy3 = -eta2*(w3+eta3*(u3+v3))+v3;
          Iij += 1.0f/sqrtf(xy1*xy1+xy2*xy2+xy3*xy3);
          Iij *= eta2;
          xy1 = eta2*(u1+w1)+eta3*v1-w1;
          xy2 = eta2*(u2+w2)+eta3*v2-w2;
          xy3 = eta2*(u3+w3)+eta3*v3-w3;
          Iij += 1.0f/sqrtf(xy1*xy1+xy2*xy2+xy3*xy3);
          I += weta2*weta3*Iij;
        }
    }

  V[tid] = J * I / F_PI / 24.0f;
}


__global__ void singleLayer1Laplace3d_commonEdge(float *us, float *vs,
                                                 float *ws, int quadn,
                                                 float *V)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int idx = 3*tid, i, j;
  float u1, u2, u3, v1, v2, v3, w1, w2, w3, J1, J2, J3, J;
  float xy1, xy2, xy3, eta2, eta3, weta2, weta3;
  float I, I11, I12, I13, I21, I22, I23, I31, I32, I33;
  
  u1 = us[idx]; u2 = us[idx+1]; u3 = us[idx+2];
  v1 = vs[idx]; v2 = vs[idx+1]; v3 = vs[idx+2];
  w1 = ws[idx]; w2 = ws[idx+1]; w3 = ws[idx+2];
  J1 = u2*v3-u3*v2; J2 = u3*v1-u1*v3; J3 = u1*v2-u2*v1;
  J = sqrtf(J1*J1+J2*J2+J3*J3);
  J1 = u2*w3-u3*w2; J2 = u3*w1-u1*w3; J3 = u1*w2-u2*w1;
  J *= sqrtf(J1*J1+J2*J2+J3*J3);
  I11 = I12 = I13 = I21 = I22 = I23 = I31 = I32 = I33 = 0.0f;

  for (i=0; i<quadn; i++)
    {
      eta2 = quadPoints[i];
      weta2 = quadWeights[i];
      for (j=0; j<quadn; j++)
        {
          eta3 = quadPoints[j];
          weta3 = quadWeights[j];
          // D2
          xy1 = eta2*(eta3*(u1+w1)-w1)+v1;
          xy2 = eta2*(eta3*(u2+w2)-w2)+v2;
          xy3 = eta2*(eta3*(u3+w3)-w3)+v3;          
          I = weta2*weta3 * eta2 / sqrtf(xy1*xy1+xy2*xy2+xy3*xy3);
          I11 += ((1.0f+eta2*eta3)/60.0f) * I;
          I12 += ((3.0f-2.0f*eta2)/120.0f) * I;
          I13 += (-eta2*(eta3-1.0f)/60.0f) * I;
          I21 += ((1.0f+2.0f*eta2*eta3)/120.0f) * I;
          I22 += ((2.0f-eta2)/60.0f) * I;
          I23 += (-eta2*(eta3-1.0f)/60.0f) * I;
          I31 += ((1.0f+3.0f*eta2*eta3)/60.0f) * I;
          I32 += ((4.0f-3.0f*eta2)/60.0f) * I;
          I33 += (-eta2*(eta3-1.0f)/20.0f) * I;
          // D3
          xy1 = -eta2*(u1+v1+eta3*w1)+v1;
          xy2 = -eta2*(u2+v2+eta3*w2)+v2;
          xy3 = -eta2*(u3+v3+eta3*w3)+v3;
          I = weta2*weta3 * eta2 / sqrtf(xy1*xy1+xy2*xy2+xy3*xy3);
          I11 += ((1.0f+eta2)/60.0f) * I;
          I12 += (1.0f/40.0f+(4.0f*eta2-eta2*eta3-3.0f*eta2*eta2*eta3)/60.0f) * I;
          I13 += (eta2*eta3*(3.0f*eta2+1.0f)/60.0f) * I;
          I21 += (1.0f/120.0f) * I;
          I22 += ((2.0f-eta2*eta3)/60.0f) * I;
          I23 += (eta2*eta3/60.0f) * I;
          I31 += ((1.0f-eta2)/60.0f) * I;
          I32 += ((3.0f*eta2*eta3-4.0f)*(eta2-1.0f)/60.0f) * I;
          I33 += (-eta2*eta3*(eta2-1.0f)/20.0f) * I;
          // D4
          xy1 = eta2*(v1-eta3*(u1+v1))-w1;
          xy2 = eta2*(v2-eta3*(u2+v2))-w2;
          xy3 = eta2*(v3-eta3*(u3+v3))-w3;
          I = weta2*weta3 * eta2 / sqrtf(xy1*xy1+xy2*xy2+xy3*xy3);
          I11 += ((1.0f+eta2*eta3)/60.0f) * I;
          I12 += ((1.0f+2.0f*eta2*eta3)/120.0f) * I;
          I13 += ((1.0f+3.0f*eta2*eta3)/60.0f) * I;
          I21 += ((3.0f-2.0f*eta2)/120.0f) * I;
          I22 += ((2.0f-eta2)/60.0f) * I;
          I23 += ((4.0f-3.0f*eta2)/60.0f) * I;
          I31 += (-eta2*(eta3-1.0f)/60.0f) * I;
          I32 += (-eta2*(eta3-1.0f)/60.0f) * I;
          I33 += (-eta2*(eta3-1.0f)/20.0f) * I;
          // D5
          xy1 = -eta2*(w1+eta3*(u1+v1))+v1;
          xy2 = -eta2*(w2+eta3*(u2+v2))+v2;
          xy3 = -eta2*(w3+eta3*(u3+v3))+v3;
          I = weta2*weta3 * eta2 / sqrtf(xy1*xy1+xy2*xy2+xy3*xy3);
          I11 += ((1.0f+eta2*eta3)/60.0f) * I;
          I12 += (1.0f/40.0f+(4.0f*eta2*eta3-eta2-3.0f*eta2*eta2*eta3)/60.0f) * I;
          I13 += (eta2*(1.0f+3.0f*eta2*eta3)/60.0f) * I;
          I21 += (1.0f/120.0f) * I;
          I22 += ((2.0f-eta2)/60.0f) * I;
          I23 += (eta2/60.0f) * I;
          I31 += ((1.0f-eta2*eta3)/60.0f) * I;
          I32 += ((3.0f*eta2-4.0f)*(eta2*eta3-1.0f)/60.0f) * I;
          I33 += (-eta2*(eta2*eta3-1.0f)/20.0f) * I;
          // D1
          xy1 = eta2*(u1+w1)+eta3*v1-w1;
          xy2 = eta2*(u2+w2)+eta3*v2-w2;
          xy3 = eta2*(u3+w3)+eta3*v3-w3;
          I = weta2*weta3 / sqrtf(xy1*xy1+xy2*xy2+xy3*xy3);
          I11 += ((1.0f+eta2)/60.0f) * I;
          I12 += (1.0f/120.0f) * I;
          I13 += ((1.0f-eta2)/60.0f) * I;
          I21 += (1.0f/40.0f+(4.0f*eta2-3.0f*eta2*eta3-eta3)/60.0f) * I;
          I22 += ((2.0f-eta3)/60.0f) * I;
          I23 += ((3.0f*eta3-4.0f)*(eta2-1.0f)/60.0f) * I;
          I31 += (eta3*(3.0f*eta2+1.0f)/60.0f) * I;
          I32 += (eta3/60.0f) * I;
          I33 += (-eta3*(eta2-1.0f)/20.0f) * I;
        }
    }

  V[9*tid] = J * I11 / F_PI / 4.0f;
  V[9*tid+1] = J * I21 / F_PI / 4.0f;
  V[9*tid+2] = J * I31 / F_PI / 4.0f;
  V[9*tid+3] = J * I12 / F_PI / 4.0f;
  V[9*tid+4] = J * I22 / F_PI / 4.0f;
  V[9*tid+5] = J * I32 / F_PI / 4.0f;
  V[9*tid+6] = J * I13 / F_PI / 4.0f;
  V[9*tid+7] = J * I23 / F_PI / 4.0f;
  V[9*tid+8] = J * I33 / F_PI / 4.0f;
}


__global__ void singleLayerLaplace3d_commonVertex(float *us, float *vs,
                                                  float *ws, float *zs,
                                                  int quadn, float *V)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int idx = 3*tid, i, j, k;
  float u1, u2, u3, v1, v2, v3, w1, w2, w3, z1, z2, z3, J1, J2, J3, J;
  float xy1, xy2, xy3, I, Iijk, eta1, eta2, eta3, weta1, weta2, weta3;
  
  u1 = us[idx]; u2 = us[idx+1]; u3 = us[idx+2];
  v1 = vs[idx]; v2 = vs[idx+1]; v3 = vs[idx+2];
  w1 = ws[idx]; w2 = ws[idx+1]; w3 = ws[idx+2];
  z1 = zs[idx]; z2 = zs[idx+1]; z3 = zs[idx+2];
  J1 = u2*v3-u3*v2; J2 = u3*v1-u1*v3; J3 = u1*v2-u2*v1;
  J = sqrtf(J1*J1+J2*J2+J3*J3);
  J1 = w2*z3-w3*z2; J2 = w3*z1-w1*z3; J3 = w1*z2-w2*z1;
  J *= sqrtf(J1*J1+J2*J2+J3*J3);

  for (i=0, I=0.0f; i<quadn; i++)
    {
      eta1 = quadPoints[i];
      weta1 = quadWeights[i];
      for (j=0; j<quadn; j++)
        {
          eta2 = quadPoints[j];
          weta2 = quadWeights[j];
          for (k=0; k<quadn; k++)
            {
              eta3 = quadPoints[k];
              weta3 = quadWeights[k];
              xy1 = u1+eta1*v1-eta2*(w1+eta3*z1);
              xy2 = u2+eta1*v2-eta2*(w2+eta3*z2);
              xy3 = u3+eta1*v3-eta2*(w3+eta3*z3);
              Iijk = 1.0f/sqrtf(xy1*xy1+xy2*xy2+xy3*xy3);
              xy1 = w1+eta1*z1-eta2*(u1+eta3*v1);
              xy2 = w2+eta1*z2-eta2*(u2+eta3*v2);
              xy3 = w3+eta1*z3-eta2*(u3+eta3*v3);
              Iijk += 1.0f/sqrtf(xy1*xy1+xy2*xy2+xy3*xy3);
              Iijk *= eta2;
              I += weta1*weta2*weta3*Iijk;
            }
        }
    }

  V[tid] = J * I / F_PI / 12.0f;
}


__global__ void singleLayer1Laplace3d_commonVertex(float *us, float *vs,
                                                   float *ws, float *zs,
                                                   int quadn, float *V)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int idx = 3*tid, i, j, k;
  float u1, u2, u3, v1, v2, v3, w1, w2, w3, z1, z2, z3, J1, J2, J3, J;
  float xy1, xy2, xy3, eta1, eta2, eta3, weta1, weta2, weta3;
  float I, I11, I12, I13, I21, I22, I23, I31, I32, I33;
  
  u1 = us[idx]; u2 = us[idx+1]; u3 = us[idx+2];
  v1 = vs[idx]; v2 = vs[idx+1]; v3 = vs[idx+2];
  w1 = ws[idx]; w2 = ws[idx+1]; w3 = ws[idx+2];
  z1 = zs[idx]; z2 = zs[idx+1]; z3 = zs[idx+2];
  J1 = u2*v3-u3*v2; J2 = u3*v1-u1*v3; J3 = u1*v2-u2*v1;
  J = sqrtf(J1*J1+J2*J2+J3*J3);
  J1 = w2*z3-w3*z2; J2 = w3*z1-w1*z3; J3 = w1*z2-w2*z1;
  J *= sqrtf(J1*J1+J2*J2+J3*J3);
  I11 = I12 = I13 = I21 = I22 = I23 = I31 = I32 = I33 = 0.0f;

  for (i=0; i<quadn; i++)
    {
      eta1 = quadPoints[i];
      weta1 = quadWeights[i];
      for (j=0; j<quadn; j++)
        {
          eta2 = quadPoints[j];
          weta2 = quadWeights[j];
          for (k=0; k<quadn; k++)
            {
              eta3 = quadPoints[k];
              weta3 = quadWeights[k];
              xy1 = u1+eta1*v1-eta2*(w1+eta3*z1);
              xy2 = u2+eta1*v2-eta2*(w2+eta3*z2);
              xy3 = u3+eta1*v3-eta2*(w3+eta3*z3);
              I = weta1*weta2*weta3*eta2 / sqrtf(xy1*xy1+xy2*xy2+xy3*xy3);
              I11 += (1.0f/12.0f-eta2/20.0f) * I;
              I12 += (eta2*eta3/20.0f) * I;
              I13 += (-eta2*(eta3-1.0f)/20.0f) * I;
              I21 += (-eta1*(4.0f*eta2-5.0f)/20.0f) * I;
              I22 += (eta1*eta2*eta3/5.0f) * I;
              I23 += (-eta1*eta2*(eta3-1.0f)/5.0f) * I;
              I31 += ((4.0f*eta2-5.0f)*(eta1-1.0f)/20.0f) * I;
              I32 += (-eta2*eta3*(eta1-1.0f)/5.0f) * I;
              I33 += (eta2*(eta1-1.0f)*(eta3-1.0f)/5.0f) * I;
              xy1 = w1+eta1*z1-eta2*(u1+eta3*v1);
              xy2 = w2+eta1*z2-eta2*(u2+eta3*v2);
              xy3 = w3+eta1*z3-eta2*(u3+eta3*v3);
              I = weta1*weta2*weta3*eta2 / sqrtf(xy1*xy1+xy2*xy2+xy3*xy3);
              I11 += (1.0f/12.0f-eta2/20.0f) * I;
              I21 += (eta2*eta3/20.0f) * I;
              I31 += (-eta2*(eta3-1.0f)/20.0f) * I;
              I12 += (-eta1*(4.0f*eta2-5.0f)/20.0f) * I;
              I22 += (eta1*eta2*eta3/5.0f) * I;
              I32 += (-eta1*eta2*(eta3-1.0f)/5.0f) * I;
              I13 += ((4.0f*eta2-5.0f)*(eta1-1.0f)/20.0f) * I;
              I23 += (-eta2*eta3*(eta1-1.0f)/5.0f) * I;
              I33 += (eta2*(eta1-1.0f)*(eta3-1.0f)/5.0f) * I;
            }
        }
    }

  V[9*tid] = J * I11 / F_PI / 4.0f;
  V[9*tid+1] = J * I21 / F_PI / 4.0f;
  V[9*tid+2] = J * I31 / F_PI / 4.0f;
  V[9*tid+3] = J * I12 / F_PI / 4.0f;
  V[9*tid+4] = J * I22 / F_PI / 4.0f;
  V[9*tid+5] = J * I32 / F_PI / 4.0f;
  V[9*tid+6] = J * I13 / F_PI / 4.0f;
  V[9*tid+7] = J * I23 / F_PI / 4.0f;
  V[9*tid+8] = J * I33 / F_PI / 4.0f;
}


__global__ void singleLayerLaplace3d_disjointPanels
(float *As, float *us, float *vs, float *Bs, float *ws, float *zs, int quadTrin,
 float *V)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int idx = 3*tid, i, j;
  float A1, A2, A3, u1, u2, u3, v1, v2, v3, B1, B2, B3, w1, w2, w3, z1, z2, z3;
  float J1, J2, J3, J;
  float x1, x2, x3, xy1, xy2, xy3, I, ksi1, ksi2, eta1, eta2, wksi, weta;

  A1 = As[idx]; A2 = As[idx+1]; A3 = As[idx+2];
  u1 = us[idx]; u2 = us[idx+1]; u3 = us[idx+2];
  v1 = vs[idx]; v2 = vs[idx+1]; v3 = vs[idx+2];
  B1 = Bs[idx]; B2 = Bs[idx+1]; B3 = Bs[idx+2];
  w1 = ws[idx]; w2 = ws[idx+1]; w3 = ws[idx+2];
  z1 = zs[idx]; z2 = zs[idx+1]; z3 = zs[idx+2];
  J1 = u2*v3-u3*v2; J2 = u3*v1-u1*v3; J3 = u1*v2-u2*v1;
  J = sqrtf(J1*J1+J2*J2+J3*J3);
  J1 = w2*z3-w3*z2; J2 = w3*z1-w1*z3; J3 = w1*z2-w2*z1;
  J *= sqrtf(J1*J1+J2*J2+J3*J3);

  for (i=0, I=0.0f; i<quadTrin; i++)
    {
      ksi1 = quadTriPoints1[i];
      ksi2 = quadTriPoints2[i];
      wksi = quadTriWeights[i];
      x1 = A1+ksi1*u1+ksi2*v1;
      x2 = A2+ksi1*u2+ksi2*v2;
      x3 = A3+ksi1*u3+ksi2*v3;
      for (j=0; j<quadTrin; j++)
        {
          eta1 = quadTriPoints1[j];
          eta2 = quadTriPoints2[j];
          weta = quadTriWeights[j];
          xy1 = x1-B1-eta1*w1-eta2*z1;
          xy2 = x2-B2-eta1*w2-eta2*z2;
          xy3 = x3-B3-eta1*w3-eta2*z3;
          
          I += wksi*weta/sqrtf(xy1*xy1+xy2*xy2+xy3*xy3);
        }
    }

  V[tid] = J * I / F_PI / 4.0f;
}


__global__ void singleLayer1Laplace3d_disjointPanels
(float *As, float *us, float *vs, float *Bs, float *ws, float *zs, int quadTrin,
 float *V)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int idx = 3*tid, i, j;
  float A1, A2, A3, u1, u2, u3, v1, v2, v3, B1, B2, B3, w1, w2, w3, z1, z2, z3;
  float J1, J2, J3, J;
  float x1, x2, x3, y1, y2, y3, xy1, xy2, xy3;
  float ksi1, ksi2, eta1, eta2, wksi, weta;
  float I, I11, I12, I13, I21, I22, I23, I31, I32, I33;

  A1 = As[idx]; A2 = As[idx+1]; A3 = As[idx+2];
  u1 = us[idx]; u2 = us[idx+1]; u3 = us[idx+2];
  v1 = vs[idx]; v2 = vs[idx+1]; v3 = vs[idx+2];
  B1 = Bs[idx]; B2 = Bs[idx+1]; B3 = Bs[idx+2];
  w1 = ws[idx]; w2 = ws[idx+1]; w3 = ws[idx+2];
  z1 = zs[idx]; z2 = zs[idx+1]; z3 = zs[idx+2];
  J1 = u2*v3-u3*v2; J2 = u3*v1-u1*v3; J3 = u1*v2-u2*v1;
  J = sqrtf(J1*J1+J2*J2+J3*J3);
  J1 = w2*z3-w3*z2; J2 = w3*z1-w1*z3; J3 = w1*z2-w2*z1;
  J *= sqrtf(J1*J1+J2*J2+J3*J3);
  I11 = I12 = I13 = I21 = I22 = I23 = I31 = I32 = I33 = 0.0f;

  for (i=0; i<quadTrin; i++)
    {
      ksi1 = quadTriPoints1[i];
      ksi2 = quadTriPoints2[i];
      wksi = quadTriWeights[i];
      x1 = A1+ksi1*u1+ksi2*v1;
      x2 = A2+ksi1*u2+ksi2*v2;
      x3 = A3+ksi1*u3+ksi2*v3;
      for (j=0; j<quadTrin; j++)
        {
          eta1 = quadTriPoints1[j];
          eta2 = quadTriPoints2[j];
          weta = quadTriWeights[j];
          y1 = B1+eta1*w1+eta2*z1;
          y2 = B2+eta1*w2+eta2*z2;
          y3 = B3+eta1*w3+eta2*z3;
          xy1 = x1-y1;
          xy2 = x2-y2;
          xy3 = x3-y3;

          I = wksi*weta/sqrtf(xy1*xy1+xy2*xy2+xy3*xy3);
          I11 += (1.0f-ksi1-ksi2)*(1.0f-eta1-eta2) * I;
          I12 += (1.0f-ksi1-ksi2)*eta1 * I;
          I13 += (1.0f-ksi1-ksi2)*eta2 * I;
          I21 += ksi1*(1.0f-eta1-eta2) * I;
          I22 += ksi1*eta1 * I;
          I23 += ksi1*eta2 * I;
          I31 += ksi2*(1.0f-eta1-eta2) * I;
          I32 += ksi2*eta1 * I;
          I33 += ksi2*eta2 * I;
        }
    }

  V[9*tid] = J * I11 / F_PI / 4.0f;
  V[9*tid+1] = J * I21 / F_PI / 4.0f;
  V[9*tid+2] = J * I31 / F_PI / 4.0f;
  V[9*tid+3] = J * I12 / F_PI / 4.0f;
  V[9*tid+4] = J * I22 / F_PI / 4.0f;
  V[9*tid+5] = J * I32 / F_PI / 4.0f;
  V[9*tid+6] = J * I13 / F_PI / 4.0f;
  V[9*tid+7] = J * I23 / F_PI / 4.0f;
  V[9*tid+8] = J * I33 / F_PI / 4.0f;
}


__global__ void doubleLayerLaplace3d_commonEdge(float *us, float *vs,
                                                float *ws, int quadn,
                                                float *K)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int idx = 3*tid, i, j;
  float u1, u2, u3, v1, v2, v3, w1, w2, w3, J1, J2, J3, J, JT, n1, n2, n3;
  float xy1, xy2, xy3, normxy, xyn, H, I1, I2, I3, I1ij, I2ij, I3ij;
  float eta1, eta2, eta3, weta2, weta3, eta12, eta123;

  u1 = us[idx]; u2 = us[idx+1]; u3 = us[idx+2];
  v1 = vs[idx]; v2 = vs[idx+1]; v3 = vs[idx+2];
  w1 = ws[idx]; w2 = ws[idx+1]; w3 = ws[idx+2];
  J1 = u2*v3-u3*v2; J2 = u3*v1-u1*v3; J3 = u1*v2-u2*v1;
  J = sqrtf(J1*J1+J2*J2+J3*J3);
  J1 = u2*w3-u3*w2; J2 = u3*w1-u1*w3; J3 = u1*w2-u2*w1;
  JT = sqrtf(J1*J1+J2*J2+J3*J3);
  n1 = J1/JT; n2 = J2/JT; n3 = J3/JT;
  J *= JT;

  eta1 = 0.5f;
  for (i=0, I1=0.0f, I2=0.0f, I3=0.0f; i<quadn; i++)
    {
      eta2 = quadPoints[i];
      weta2 = quadWeights[i];
      eta12 = eta1*eta2;
      for (j=0; j<quadn; j++)
        {
          eta3 = quadPoints[j];
          weta3 = quadWeights[j];
          eta123 = eta12*eta3;
          xy1 = eta2*(eta3*(u1+w1)-w1)+v1;
          xy2 = eta2*(eta3*(u2+w2)-w2)+v2;
          xy3 = eta2*(eta3*(u3+w3)-w3)+v3;
          normxy = sqrtf(xy1*xy1+xy2*xy2+xy3*xy3);
          xyn = xy1*n1+xy2*n2+xy3*n3;
          H = xyn/normxy/normxy/normxy;
          I1ij = quadWeights2[0]*quadPoints2[0]
            * (1.0f-quadPoints2[0]*(1.0f-eta123));
          I1ij+= quadWeights2[1]*quadPoints2[1]
            * (1.0f-quadPoints2[1]*(1.0f-eta123));
          I1ij *= H;
          I2ij = quadWeights2[0]*quadPoints2[0] * quadPoints2[0]
            * (1.0f-eta12);
          I2ij += quadWeights2[1]*quadPoints2[1] * quadPoints2[1]*(1.0f-eta12);
          I2ij *= H;
          I3ij = quadWeights2[0]*quadPoints2[0] * quadPoints2[0]*eta12
            *(1.0f-eta3);
          I3ij += quadWeights2[1]*quadPoints2[1] * quadPoints2[1]*eta12
            * (1.0f-eta3);
          I3ij *= H;
      
          xy1 = -eta2*(u1+v1+eta3*w1)+v1;
          xy2 = -eta2*(u2+v2+eta3*w2)+v2;
          xy3 = -eta2*(u3+v3+eta3*w3)+v3;
          normxy = sqrtf(xy1*xy1+xy2*xy2+xy3*xy3);
          xyn = xy1*n1+xy2*n2+xy3*n3;
          H = xyn/normxy/normxy/normxy;
          I1ij += quadWeights2[0]*quadPoints2[0] * (1.0f-quadPoints2[0]) * H;
          I1ij += quadWeights2[1]*quadPoints2[1] * (1.0f-quadPoints2[1]) * H;
          I2ij += quadWeights2[0]*quadPoints2[0] * quadPoints2[0]
            * (1.0f-eta123) * H;
          I2ij += quadWeights2[1]*quadPoints2[1] * quadPoints2[1]
            * (1.0f-eta123) * H;
          I3ij += quadWeights2[0]*quadPoints2[0] * quadPoints2[0]*eta123 * H;
          I3ij += quadWeights2[1]*quadPoints2[1] * quadPoints2[1]*eta123 * H;
      
          xy1 = eta2*(v1-eta3*(u1+v1))-w1;
          xy2 = eta2*(v2-eta3*(u2+v2))-w2;
          xy3 = eta2*(v3-eta3*(u3+v3))-w3;
          normxy = sqrtf(xy1*xy1+xy2*xy2+xy3*xy3);
          xyn = xy1*n1+xy2*n2+xy3*n3;
          H = xyn/normxy/normxy/normxy;
          I1ij += quadWeights2[0]*quadPoints2[0] * (1.0f-quadPoints2[0]) * H;
          I1ij += quadWeights2[1]*quadPoints2[1] * (1.0f-quadPoints2[1]) * H;
          I2ij += quadWeights2[0]*quadPoints2[0] * quadPoints2[0]
            * (1.0f-eta1) * H;
          I2ij += quadWeights2[1]*quadPoints2[1] * quadPoints2[1]
            * (1.0f-eta1) * H;
          I3ij += quadWeights2[0]*quadPoints2[0] * quadPoints2[0]*eta1 * H;
          I3ij += quadWeights2[1]*quadPoints2[1] * quadPoints2[1]*eta1 * H;
      
          xy1 = -eta2*(w1+eta3*(u1+v1))+v1;
          xy2 = -eta2*(w2+eta3*(u2+v2))+v2;
          xy3 = -eta2*(w3+eta3*(u3+v3))+v3;
          normxy = sqrtf(xy1*xy1+xy2*xy2+xy3*xy3);
          xyn = xy1*n1+xy2*n2+xy3*n3;
          H = xyn/normxy/normxy/normxy;
          I1ij += quadWeights2[0]*quadPoints2[0] * (1.0f-quadPoints2[0]) * H;
          I1ij += quadWeights2[1]*quadPoints2[1] * (1.0f-quadPoints2[1]) * H;
          I2ij += quadWeights2[0]*quadPoints2[0] * quadPoints2[0]
            * (1.0f-eta12) * H;
          I2ij += quadWeights2[1]*quadPoints2[1] * quadPoints2[1]
            * (1.0f-eta12) * H;
          I3ij += quadWeights2[0]*quadPoints2[0] * quadPoints2[0]*eta12 * H;
          I3ij += quadWeights2[1]*quadPoints2[1] * quadPoints2[1]*eta12 * H;
      
          I1ij *= eta2;
          I2ij *= eta2;
          I3ij *= eta2;
      
          xy1 = eta2*(u1+w1)+eta3*v1-w1;
          xy2 = eta2*(u2+w2)+eta3*v2-w2;
          xy3 = eta2*(u3+w3)+eta3*v3-w3;
          normxy = sqrtf(xy1*xy1+xy2*xy2+xy3*xy3);
          xyn = xy1*n1+xy2*n2+xy3*n3;
          H = xyn/normxy/normxy/normxy;
          I1ij += quadWeights2[0]*quadPoints2[0]
            * (1.0f-quadPoints2[0]*(1.0f-eta12)) * H;
          I1ij += quadWeights2[1]*quadPoints2[1]
            * (1.0f-quadPoints2[1]*(1.0f-eta12)) * H;
          I2ij += quadWeights2[0]*quadPoints2[0]
            * quadPoints2[0]*(1.0f-eta1) * H;
          I2ij += quadWeights2[1]*quadPoints2[1]
            * quadPoints2[1]*(1.0f-eta1) * H;
          I3ij += quadWeights2[0]*quadPoints2[0]
            * quadPoints2[0]*eta1*(1.0f-eta2) * H;
          I3ij += quadWeights2[1]*quadPoints2[1]
            * quadPoints2[1]*eta1*(1.0f-eta2) * H;

          I1 += weta2*weta3*I1ij;
          I2 += weta2*weta3*I2ij;
          I3 += weta2*weta3*I3ij;
        }
    }

  K[idx] = J * I1 / 4.0f / F_PI;
  K[idx+1] = J * I2 / 4.0f / F_PI;
  K[idx+2] = J * I3 / 4.0f / F_PI;
}


__global__ void doubleLayerLaplace3d_commonVertex(float *us, float *vs,
                                                  float *ws, float *zs,
                                                  int quadn, float *K)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int idx = 3*tid, i, j, k;
  float u1, u2, u3, v1, v2, v3, w1, w2, w3, z1, z2, z3;
  float J1, J2, J3, J, JT, n1, n2, n3;
  float xy1, xy2, xy3, normxy, xyn, H, I1, I2, I3, I1ijk, I2ijk, I3ijk;
  float eta1, eta2, eta3, weta1, weta2, weta3;
  
  u1 = us[idx]; u2 = us[idx+1]; u3 = us[idx+2];
  v1 = vs[idx]; v2 = vs[idx+1]; v3 = vs[idx+2];
  w1 = ws[idx]; w2 = ws[idx+1]; w3 = ws[idx+2];
  z1 = zs[idx]; z2 = zs[idx+1]; z3 = zs[idx+2];
  J1 = u2*v3-u3*v2; J2 = u3*v1-u1*v3; J3 = u1*v2-u2*v1;
  J = sqrtf(J1*J1+J2*J2+J3*J3);
  J1 = w2*z3-w3*z2; J2 = w3*z1-w1*z3; J3 = w1*z2-w2*z1;
  JT = sqrtf(J1*J1+J2*J2+J3*J3);
  n1 = J1/JT; n2 = J2/JT; n3 = J3/JT;
  J *= JT;

  for (i=0, I1=0.0f, I2=0.0f, I3=0.0f; i<quadn; i++)
    {
      eta1 = quadPoints[i];
      weta1 = quadWeights[i];
      for (j=0; j<quadn; j++)
        {
          eta2 = quadPoints[j];
          weta2 = quadWeights[j];
          for (k=0; k<quadn; k++)
            {
              eta3 = quadPoints[k];
              weta3 = quadWeights[k];
              xy1 = u1+eta1*v1-eta2*(w1+eta3*z1);
              xy2 = u2+eta1*v2-eta2*(w2+eta3*z2);
              xy3 = u3+eta1*v3-eta2*(w3+eta3*z3);
              normxy = sqrtf(xy1*xy1+xy2*xy2+xy3*xy3);
              xyn = xy1*n1+xy2*n2+xy3*n3;
              H = xyn/normxy/normxy/normxy;
              I1ijk = quadWeights2[0]*quadPoints2[0]*(1.0f-quadPoints2[0]*eta2);
              I1ijk += quadWeights2[1]*quadPoints2[1]
                * (1.0f-quadPoints2[1]*eta2);
              I1ijk *= H;
              I2ijk = quadWeights2[0]*quadPoints2[0]*quadPoints2[0]
                * eta2*(1.0f-eta3);
              I2ijk += quadWeights2[1]*quadPoints2[1]*quadPoints2[1]
                * eta2*(1.0f-eta3);
              I2ijk *= H;
              I3ijk = quadWeights2[0]*quadPoints2[0]*quadPoints2[0]*eta2*eta3;
              I3ijk += quadWeights2[1]*quadPoints2[1]*quadPoints2[1]*eta2*eta3;
              I3ijk *= H;
	
              xy1 = eta2*(u1+eta3*v1)-(w1+eta1*z1);
              xy2 = eta2*(u2+eta3*v2)-(w2+eta1*z2);
              xy3 = eta2*(u3+eta3*v3)-(w3+eta1*z3);
              normxy = sqrtf(xy1*xy1+xy2*xy2+xy3*xy3);
              xyn = xy1*n1+xy2*n2+xy3*n3;
              H = xyn/normxy/normxy/normxy;
              I1ijk += quadWeights2[0]*quadPoints2[0]
                * (1.0f-quadPoints2[0]) * H;
              I1ijk += quadWeights2[1]*quadPoints2[1]
                * (1.0f-quadPoints2[1]) * H;
              I2ijk += quadWeights2[0]*quadPoints2[0]*quadPoints2[0]
                * (1.0f-eta1) * H;
              I2ijk += quadWeights2[1]*quadPoints2[1]*quadPoints2[1]
                * (1.0f-eta1) * H;
              I3ijk += quadWeights2[0]*quadPoints2[0]*quadPoints2[0]*eta1 * H;
              I3ijk += quadWeights2[1]*quadPoints2[1]*quadPoints2[1]*eta1 * H;
	
              I1 += weta1*weta2*weta3 * eta2 * I1ijk;
              I2 += weta1*weta2*weta3 * eta2 * I2ijk;
              I3 += weta1*weta2*weta3 * eta2 * I3ijk;
            }
        }
    }

  K[idx] = J * I1 / 4.0f / F_PI;
  K[idx+1] = J * I2 / 4.0f / F_PI;
  K[idx+2] = J * I3 / 4.0f / F_PI;
}


__global__ void doubleLayerLaplace3d_disjointPanels
(float *As, float *us, float *vs, float *Bs, float *ws, float *zs, int quadTrin,
 float *K)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int idx = 3*tid, i, j;
  float A1, A2, A3, u1, u2, u3, v1, v2, v3, B1, B2, B3, w1, w2, w3, z1, z2, z3;
  float J1, J2, J3, J, JT, n1, n2, n3;
  float x1, x2, x3, xy1, xy2, xy3, normxy, xyn, H, I1, I2, I3;
  float ksi1, ksi2, eta1, eta2, wksi, weta;

  A1 = As[idx]; A2 = As[idx+1]; A3 = As[idx+2];
  u1 = us[idx]; u2 = us[idx+1]; u3 = us[idx+2];
  v1 = vs[idx]; v2 = vs[idx+1]; v3 = vs[idx+2];
  B1 = Bs[idx]; B2 = Bs[idx+1]; B3 = Bs[idx+2];
  w1 = ws[idx]; w2 = ws[idx+1]; w3 = ws[idx+2];
  z1 = zs[idx]; z2 = zs[idx+1]; z3 = zs[idx+2];
  J1 = u2*v3-u3*v2; J2 = u3*v1-u1*v3; J3 = u1*v2-u2*v1;
  J = sqrtf(J1*J1+J2*J2+J3*J3);
  J1 = w2*z3-w3*z2; J2 = w3*z1-w1*z3; J3 = w1*z2-w2*z1;
  JT = sqrtf(J1*J1+J2*J2+J3*J3);
  n1 = J1/JT; n2 = J2/JT; n3 = J3/JT;
  J *= JT;

  for (i=0, I1=0.0f, I2=0.0f, I3=0.0f; i<quadTrin; i++)
    {
      ksi1 = quadTriPoints1[i];
      ksi2 = quadTriPoints2[i];
      wksi = quadTriWeights[i];
      x1 = A1+ksi1*u1+ksi2*v1;
      x2 = A2+ksi1*u2+ksi2*v2;
      x3 = A3+ksi1*u3+ksi2*v3;
      for (j=0; j<quadTrin; j++)
        {
          eta1 = quadTriPoints1[j];
          eta2 = quadTriPoints2[j];
          weta = quadTriWeights[j];
          xy1 = x1-B1-eta1*w1-eta2*z1;
          xy2 = x2-B2-eta1*w2-eta2*z2;
          xy3 = x3-B3-eta1*w3-eta2*z3;
          normxy = sqrtf(xy1*xy1+xy2*xy2+xy3*xy3);
          xyn = xy1*n1+xy2*n2+xy3*n3;
          H = xyn/normxy/normxy/normxy;
          I1 += wksi*weta * (1.0f-eta1-eta2) * H;
          I2 += wksi*weta * eta1 * H;
          I3 += wksi*weta * eta2 * H;
        }
    }

  K[idx] = J * I1 / F_PI / 4.0f;
  K[idx+1] = J * I2 / F_PI / 4.0f;
  K[idx+2] = J * I3 / F_PI / 4.0f;
}


void initGPU (int annodes, const double *anodes,
              int ane, const int *anelemNodes, int anorder,
              int anblocks, int anthreads)
{
  int i;
  
  // batch size
  nblocks = anblocks;
  nthreads = anthreads;
  nn = nblocks*nthreads;

  // CPU-from/to-GPU memory
  As = new float[3*nn];
  us = new float[3*nn];
  vs = new float[3*nn];
  Bs = new float[3*nn];
  ws = new float[3*nn];
  zs = new float[3*nn];
  for (int i=0; i<3*nn; i++)
    {
      As[i] = -10.0f + (float)rand()/(float)(RAND_MAX/2.0f);
      us[i] = -1.0f + (float)rand()/(float)(RAND_MAX/2.0f);
      vs[i] = -1.0f + (float)rand()/(float)(RAND_MAX/2.0f);
      Bs[i] = 10.0f + (float)rand()/(float)(RAND_MAX/2.0f);
      ws[i] = -1.0f + (float)rand()/(float)(RAND_MAX/2.0f);
      zs[i] = -1.0f + (float)rand()/(float)(RAND_MAX/2.0f);
    }
  Matrix = new float[9*nn];
  memset(Matrix,0,3*nn*sizeof(float));
  cudaMalloc(&dev_As,3*nn*sizeof(float));
  cudaMalloc(&dev_us,3*nn*sizeof(float));
  cudaMalloc(&dev_vs,3*nn*sizeof(float));
  cudaMalloc(&dev_Bs,3*nn*sizeof(float));
  cudaMalloc(&dev_ws,3*nn*sizeof(float));
  cudaMalloc(&dev_zs,3*nn*sizeof(float));
  cudaMalloc(&dev_Matrix,9*nn*sizeof(float));

  // Store Gauss quadratures to GPU shared memory
  quadn = LineQuadratureSizes[anorder];
  quadTrin = TriangleQuadratureSizes[anorder];
  quPoints2 = new float[2];
  quWeights2 = new float[2];
  for (i=0; i<2; i++)
    {
      quPoints2[i] =  (float)(LineQuadraturePoints[1][i]);
      quWeights2[i] =  (float)(LineQuadratureWeights[1][i]);
    }
  quPoints3 = new float[3];
  quWeights3 = new float[3];
  for (i=0; i<3; i++)
    {
      quPoints3[i] =  (float)(LineQuadraturePoints[2][i]);
      quWeights3[i] =  (float)(LineQuadratureWeights[2][i]);
    }
  quPoints4 = new float[4];
  quWeights4 = new float[4];
  for (i=0; i<4; i++)
    {
      quPoints4[i] =  (float)(LineQuadraturePoints[3][i]);
      quWeights4[i] =  (float)(LineQuadratureWeights[3][i]);
    }
  quPoints = new float[quadn];
  quWeights = new float[quadn];
  for (i=0; i<quadn; i++)
  {
    quPoints[i] = (float)(LineQuadraturePoints[anorder][i]);
    quWeights[i] = (float)(LineQuadratureWeights[anorder][i]);
  }
  quTriPoints1 = new float[quadTrin];
  quTriPoints2 = new float[quadTrin];
  quTriWeights = new float[quadTrin];
  for (i=0; i<quadTrin; i++)
  {
    quTriPoints1[i] = (float)(TriangleQuadraturePoints[anorder][i].x);
    quTriPoints2[i] = (float)(TriangleQuadraturePoints[anorder][i].y);
    quTriWeights[i] = (float)(TriangleQuadratureWeights[anorder][i]);
  }
  cudaMemcpyToSymbol(quadPoints2,quPoints2,2*sizeof(float));
  cudaMemcpyToSymbol(quadWeights2,quWeights2,2*sizeof(float));
  cudaMemcpyToSymbol(quadPoints3,quPoints3,3*sizeof(float));
  cudaMemcpyToSymbol(quadWeights3,quWeights3,3*sizeof(float));
  cudaMemcpyToSymbol(quadPoints4,quPoints4,4*sizeof(float));
  cudaMemcpyToSymbol(quadWeights4,quWeights4,4*sizeof(float));
  cudaMemcpyToSymbol(quadPoints,quPoints,quadn*sizeof(float));
  cudaMemcpyToSymbol(quadWeights,quWeights,quadn*sizeof(float));
  cudaMemcpyToSymbol(quadTriPoints1,quTriPoints1,quadTrin*sizeof(float));
  cudaMemcpyToSymbol(quadTriPoints2,quTriPoints2,quadTrin*sizeof(float));
  cudaMemcpyToSymbol(quadTriWeights,quTriWeights,quadTrin*sizeof(float));

  // Mesh topology
  createMeshTopology(annodes,anodes,ane,anelemNodes);
}


void closeGPU ()
{  
  // Release CPU-from/to-GPU memory
  delete [] As;
  delete [] us;
  delete [] vs;
  delete [] Bs;
  delete [] ws;
  delete [] zs;
  delete [] Matrix;
  cudaFree(dev_As);
  cudaFree(dev_us);
  cudaFree(dev_vs);
  cudaFree(dev_Bs);
  cudaFree(dev_ws);
  cudaFree(dev_zs);
  cudaFree(dev_Matrix);

  // Release Gauss quadratures
  delete [] quPoints2;
  delete [] quWeights2;
  delete [] quPoints3;
  delete [] quWeights3;
  delete [] quPoints4;
  delete [] quWeights4;
  delete [] quPoints;
  delete [] quWeights;
  delete [] quTriPoints1;
  delete [] quTriPoints2;
  delete [] quTriWeights;

  // Release mesh topology
  releaseMeshTopology();
}


int getPanelsCase (int *elemNodes1, int *elemEdges1,
                   int *elemNodes2, int *elemEdges2)
{
  int i, j, ei, pi;
  if (elemNodes1[0]==elemNodes2[0] && elemNodes1[1]==elemNodes2[1] &&
      elemNodes1[2]==elemNodes2[2])
    return 0;
  for (i=0; i<3; i++)
    {
      ei = abs(elemEdges1[i]);
      for (j=0; j<3; j++)
        if (ei==abs(elemEdges2[j]))
          return 1;
    }
  for (i=0; i<3; i++)
    {
      pi = elemNodes1[i];
      for (j=0; j<3; j++)
        if (pi==elemNodes2[j])
          return 2;
    }
  return 3;
}


void applyGPULaplace3d1Layer (const double *vec, double *result, int ansatz)
{
  int i, j, k, l, idx, idx1, idx2;

  int progress = 0;
  std::cout << "BEM 3d Laplace GPU-apply 1-layer - " << nnodes << " nodes, "
            << ne << " elements: " << progress << "%" << std::flush;
  if (ansatz==0)
    memset(result,0,ne*sizeof(double));
  else // ansatz==1
    memset(result,0,nnodes*sizeof(double));
    
  // Assemble identical-panels entries
  int *elemNodes1;
  int ni;
  double P1[3], P2[3], P3[3], Q1[3], Q2[3], Q3[3];
  int fi, fiTimes3, fj;
  int fis[nn], fjs[nn], ks[nn], ls[nn];
  for (i=0; i<=(ne-1)/nn+1; i++)
    {
      idx1 = i*nn;
      idx2 = mymin<int>(ne,(i+1)*nn);
      ni = idx2-idx1;
      for (j=idx1, idx=0; j<idx2; j++, idx++)
        {
          elemNodes1 = elemNodes+3*j;
          memcpy(P1,nodes+3*(elemNodes1[0]-1),3*sizeof(double));
          memcpy(P2,nodes+3*(elemNodes1[1]-1),3*sizeof(double));
          memcpy(P3,nodes+3*(elemNodes1[2]-1),3*sizeof(double));
          for (k=0; k<3; k++)
            {
              us[3*idx+k] = (float) (P2[k]-P1[k]);
              vs[3*idx+k] = (float) (P3[k]-P2[k]);
            }
          fis[idx] = j+1;
        }
      cudaMemcpy(dev_us,us,3*ni*sizeof(float),cudaMemcpyHostToDevice);
      cudaMemcpy(dev_vs,vs,3*ni*sizeof(float),cudaMemcpyHostToDevice);
      if (ansatz==0)
        {
          singleLayerLaplace3d_idPanels<<<nblocks,nthreads>>>(dev_us,dev_vs,
                                                              quadn,
                                                              dev_Matrix);
          cudaMemcpy(Matrix,dev_Matrix,ni*sizeof(float),cudaMemcpyDeviceToHost);
          for (j=idx1, idx=0; j<idx2; j++, idx++)
            result[j] += ((double)(Matrix[idx])) * vec[j];
        }
      else // ansatz==1
        {
          singleLayer1Laplace3d_idPanels<<<nblocks,nthreads>>>(dev_us,dev_vs,
                                                               quadn,
                                                               dev_Matrix);
          cudaMemcpy(Matrix,dev_Matrix,9*ni*sizeof(float),
                     cudaMemcpyDeviceToHost);
          for (idx=0; idx<ni; idx++)
            {
              elemNodes1 = elemNodes+3*(fis[idx]-1);
              for (k=0; k<3; k++)
                for (l=0; l<3; l++)
                  result[elemNodes1[k]-1] += ((double)(Matrix[9*idx+3*l+k]))
                    * vec[elemNodes1[l]-1];
            }
        }
    }

  // Assemble common-edge entries
  int *elemNodes2, *elemEdges1, *elemEdges2;
  int kk;
  fiTimes3 = 0;
  for (i=0; i<=(3*ne-1)/nn+1; i++)
    {
      idx1 = i*nn;
      idx2 = mymin<int>(3*ne,(i+1)*nn);
      ni = idx2-idx1;
      for (j=idx1, idx=0; j<idx2; j++, idx++, fiTimes3++)
        {
          fi = fiTimes3/3+1;
          elemNodes1 = elemNodes+3*(fi-1);
          elemEdges1 = elemEdges+3*(fi-1);
          fj = commEdgeElems2Elems[j];
          elemNodes2 = elemNodes+3*(fj-1);
          elemEdges2 = elemEdges+3*(fj-1);
          for (k=0; k<3; k++)
            {
              l = Find<int>(-elemEdges1[k],elemEdges2,3);
              if (l>0)
                {
                  l = l-1;
                  memcpy(P1,nodes+3*(elemNodes1[(k+1)%3]-1),3*sizeof(double));
                  memcpy(P2,nodes+3*(elemNodes1[k]-1),3*sizeof(double));
                  memcpy(P3,nodes+3*(elemNodes1[(k+2)%3]-1),3*sizeof(double));
                  memcpy(Q1,nodes+3*(elemNodes2[l]-1),3*sizeof(double));
                  memcpy(Q2,nodes+3*(elemNodes2[(l+1)%3]-1),3*sizeof(double));
                  memcpy(Q3,nodes+3*(elemNodes2[(l+2)%3]-1),3*sizeof(double));
                  ks[idx] = k;
                  ls[idx] = l;
                 break;
                }
            }
          for (k=0; k<3; k++)
            {
              us[3*idx+k] = (float) (P2[k]-P1[k]);
              vs[3*idx+k] = (float) (P3[k]-P2[k]);
              ws[3*idx+k] = (float) (Q3[k]-Q2[k]);
            }
          fis[idx] = fiTimes3/3+1;
          fjs[idx] = fj;
        }
      cudaMemcpy(dev_us,us,3*ni*sizeof(float),cudaMemcpyHostToDevice);
      cudaMemcpy(dev_vs,vs,3*ni*sizeof(float),cudaMemcpyHostToDevice);
      cudaMemcpy(dev_ws,ws,3*ni*sizeof(float),cudaMemcpyHostToDevice);
      if (ansatz==0)
        {
          singleLayerLaplace3d_commonEdge<<<nblocks,nthreads>>>(dev_us,dev_vs,
                                                                dev_ws,
                                                                quadn,
                                                                dev_Matrix);
          cudaMemcpy(Matrix,dev_Matrix,ni*sizeof(float),cudaMemcpyDeviceToHost);
          for (idx=0; idx<ni; idx++)
            result[fis[idx]-1] += ((double)(Matrix[idx])) * vec[fjs[idx]-1];
        }
      else // ansatz==1
        {
          singleLayer1Laplace3d_commonEdge<<<nblocks,nthreads>>>(dev_us,dev_vs,
                                                                 dev_ws,
                                                                 quadn,
                                                                 dev_Matrix);
          cudaMemcpy(Matrix,dev_Matrix,9*ni*sizeof(float),
                     cudaMemcpyDeviceToHost);
          for (idx=0; idx<ni; idx++)
            {
              elemNodes1 = elemNodes+3*(fis[idx]-1);
              elemNodes2 = elemNodes+3*(fjs[idx]-1);
              for (k=0; k<3; k++)
                {
                  if (k==0)
                    kk = elemNodes1[(ks[idx]+1)%3]-1;
                  if (k==1)
                    kk = elemNodes1[ks[idx]]-1;
                  if (k==2)
                    kk = elemNodes1[(ks[idx]+2)%3];
                  for (l=0; l<3; l++)
                    result[kk] += ((double)(Matrix[9*idx+3*l+k]))
                      * vec[elemNodes1[(ls[idx]+l)%3]-1];
                }
            }
        }
    }

  // Assemble common-vertex entries
  fi = 1;
  int fiIdx = 0;
  for (i=0; i<=(commNodesNTotal-1)/nn+1; i++)
    {
      idx1 = i*nn;
      idx2 = mymin<int>(commNodesNTotal,(i+1)*nn);
      ni = idx2-idx1;
      for (j=idx1, idx=0; j<idx2; j++, idx++)
        {
          while (commNodeElems2ElemsSizes[fi-1]==0)
            {
              fi++;
              fiIdx = 0;
            }
          elemNodes1 = elemNodes+3*(fi-1);
          elemEdges1 = elemEdges+3*(fi-1);
          fj = commNodeElems2Elems[commNodeElems2ElemsIdx[fi-1]+fiIdx];
          elemNodes2 = elemNodes+3*(fj-1);
          elemEdges2 = elemEdges+3*(fj-1);
          for (k=0; k<3; k++)
            {
              l = Find<int>(elemNodes1[k],elemNodes2,3);
              if (l>0)
                {
                  l = l-1;
                  memcpy(P1,nodes+3*(elemNodes1[k]-1),3*sizeof(double));
                  memcpy(P2,nodes+3*(elemNodes1[(k+1)%3]-1),3*sizeof(double));
                  memcpy(P3,nodes+3*(elemNodes1[(k+2)%3]-1),3*sizeof(double));
                  memcpy(Q1,nodes+3*(elemNodes2[l]-1),3*sizeof(double));
                  memcpy(Q2,nodes+3*(elemNodes2[(l+1)%3]-1),3*sizeof(double));
                  memcpy(Q3,nodes+3*(elemNodes2[(l+2)%3]-1),3*sizeof(double));
                  ks[idx] = k;
                  ls[idx] = l;
                  break;
                }
            }
          for (k=0; k<3; k++)
            {
              us[3*idx+k] = (float) (P2[k]-P1[k]);
              vs[3*idx+k] = (float) (P3[k]-P2[k]);
              ws[3*idx+k] = (float) (Q2[k]-Q1[k]);
              zs[3*idx+k] = (float) (Q3[k]-Q2[k]);
            }
          fis[idx] = fi;
          fjs[idx] = fj;
          if (fiIdx+1<commNodeElems2ElemsSizes[fi-1])
            fiIdx++;
          else
            {
              fi++;
              fiIdx = 0;
            }
        }
      cudaMemcpy(dev_us,us,3*ni*sizeof(float),cudaMemcpyHostToDevice);
      cudaMemcpy(dev_vs,vs,3*ni*sizeof(float),cudaMemcpyHostToDevice);
      cudaMemcpy(dev_ws,ws,3*ni*sizeof(float),cudaMemcpyHostToDevice);
      cudaMemcpy(dev_zs,zs,3*ni*sizeof(float),cudaMemcpyHostToDevice);
      if (ansatz==0)
        {
          singleLayerLaplace3d_commonVertex<<<nblocks,nthreads>>>(dev_us,dev_vs,
                                                                  dev_ws,dev_zs,
                                                                  quadn,
                                                                  dev_Matrix);
          cudaMemcpy(Matrix,dev_Matrix,ni*sizeof(float),cudaMemcpyDeviceToHost);
          for (idx=0; idx<ni; idx++)
            result[fis[idx]-1] += ((double)(Matrix[idx])) * vec[fjs[idx]-1];
        }
      else // ansatz==1
        {
          singleLayer1Laplace3d_commonVertex<<<nblocks,nthreads>>>(dev_us,
                                                                   dev_vs,
                                                                  dev_ws,dev_zs,
                                                                  quadn,
                                                                  dev_Matrix);
          cudaMemcpy(Matrix,dev_Matrix,9*ni*sizeof(float),
                     cudaMemcpyDeviceToHost);
          for (idx=0; idx<ni; idx++)
            {
              elemNodes1 = elemNodes+3*(fis[idx]-1);
              elemNodes2 = elemNodes+3*(fjs[idx]-1);
              for (k=0; k<3; k++)
                for (l=0; l<3; l++)
                  result[elemNodes1[(ks[idx]+k)%3]-1] +=
                    ((double)(Matrix[9*idx+3*l+k]))
                    * vec[elemNodes1[(ls[idx]+l)%3]-1];
            }
        }
    }
  
  // Assemble disjoint-panels entries
  fi = 1; fj = 1;
  long int iidx1, iidx2;
  for (i=0; i<=(disjPanelsNTotal-1)/nn+1; i++)
    {
      iidx1 = i*((long int)nn);
      iidx2 = mymin<long int>(disjPanelsNTotal,(i+1)*((long int)nn));
      ni = ((int)(iidx2-iidx1));
      for (idx=0; idx<ni; idx++)
        {
          while (1)
            {
              // skip identical panels
              if (fi==fj)
                {
                  nextIteration(fi,fj,1,ne);
                  continue;
                }
              // skip common-edge panels
              if (commEdgeElems2Elems[3*(fi-1)]==fj ||
                  commEdgeElems2Elems[3*(fi-1)+1]==fj ||
                  commEdgeElems2Elems[3*(fi-1)+2]==fj)
                {
                  nextIteration(fi,fj,1,ne);
                  continue;
                }
              // skip common-vertex panels
              for (k=0; k<commNodeElems2ElemsSizes[fi-1]; k++)
                if (commNodeElems2Elems[commNodeElems2ElemsIdx[fi-1]+k]==fj)
                  break;
              if (k<commNodeElems2ElemsSizes[fi-1])
                {
                  nextIteration(fi,fj,1,ne);
                  continue;
                }
              break;
            }
          elemNodes1 = elemNodes+3*(fi-1);
          elemEdges1 = elemEdges+3*(fi-1);
          elemNodes2 = elemNodes+3*(fj-1);
          elemEdges2 = elemEdges+3*(fj-1);
          memcpy(P1,nodes+3*(elemNodes1[0]-1),3*sizeof(double));
          memcpy(P2,nodes+3*(elemNodes1[1]-1),3*sizeof(double));
          memcpy(P3,nodes+3*(elemNodes1[2]-1),3*sizeof(double));
          memcpy(Q1,nodes+3*(elemNodes2[0]-1),3*sizeof(double));
          memcpy(Q2,nodes+3*(elemNodes2[1]-1),3*sizeof(double));
          memcpy(Q3,nodes+3*(elemNodes2[2]-1),3*sizeof(double));
          for (k=0; k<3; k++)
            {
              As[3*idx+k] = (float) (P1[k]);
              us[3*idx+k] = (float) (P2[k]-P1[k]);
              vs[3*idx+k] = (float) (P3[k]-P1[k]);
              Bs[3*idx+k] = (float) (Q1[k]);
              ws[3*idx+k] = (float) (Q2[k]-Q1[k]);
              zs[3*idx+k] = (float) (Q3[k]-Q1[k]);
            }
          fis[idx] = fi;
          fjs[idx] = fj;
          nextIteration(fi,fj,1,ne);
        }
      cudaMemcpy(dev_As,As,3*ni*sizeof(float),cudaMemcpyHostToDevice);
      cudaMemcpy(dev_us,us,3*ni*sizeof(float),cudaMemcpyHostToDevice);
      cudaMemcpy(dev_vs,vs,3*ni*sizeof(float),cudaMemcpyHostToDevice);
      cudaMemcpy(dev_Bs,Bs,3*ni*sizeof(float),cudaMemcpyHostToDevice);
      cudaMemcpy(dev_ws,ws,3*ni*sizeof(float),cudaMemcpyHostToDevice);
      cudaMemcpy(dev_zs,zs,3*ni*sizeof(float),cudaMemcpyHostToDevice);
      if (ansatz==0)
        {
          singleLayerLaplace3d_disjointPanels<<<nblocks,nthreads>>>
            (dev_As,dev_us,dev_vs,dev_Bs,dev_ws,dev_zs,quadTrin,dev_Matrix);
          cudaMemcpy(Matrix,dev_Matrix,ni*sizeof(float),cudaMemcpyDeviceToHost);
          for (idx=0; idx<ni; idx++)
            result[fis[idx]-1] += ((double)(Matrix[idx])) * vec[fjs[idx]-1];
        }
      else // ansatz==1
        {
          singleLayer1Laplace3d_disjointPanels<<<nblocks,nthreads>>>
            (dev_As,dev_us,dev_vs,dev_Bs,dev_ws,dev_zs,quadTrin,dev_Matrix);
          cudaMemcpy(Matrix,dev_Matrix,9*ni*sizeof(float),
                     cudaMemcpyDeviceToHost);
           for (idx=0; idx<ni; idx++)
            {
              elemNodes1 = elemNodes+3*(fis[idx]-1);
              elemNodes2 = elemNodes+3*(fjs[idx]-1);
              for (k=0; k<3; k++)
                for (l=0; l<3; l++)
                  result[elemNodes1[k]-1] +=
                    ((double)(Matrix[9*idx+3*l+k])) * vec[elemNodes1[l]-1];
            }
        }
      progress = (100*i)/ne;
      std::cout << "\rBEM 3d Laplace GPU-apply 1-layer - "
                << nnodes << " nodes, "
                << ne << " elements: " << progress << "%" << std::flush;
    }
  std::cout << "\rBEM 3d Laplace GPU-apply 1-layer - " << nnodes << " nodes, "
            << ne << " elements: 100%" << std::flush;
  std::cout << std::endl;
}


void applyGPULaplace3d2Layer (const double *vec, double *result)
{
  int i, j, k, l, idx, idx1, idx2;

  int progress = 0;
  std::cout << "BEM 3d Laplace GPU-apply 2-layer - " << nnodes << " nodes, "
            << ne << " elements: " << progress << "%" << std::flush;
  memset(result,0,ne*sizeof(double));
    
  // Assemble common-edge entries
  int *elemNodes1;
  int ni;
  double P1[3], P2[3], P3[3], Q1[3], Q2[3], Q3[3];
  int fi, fiTimes3, fj;
  int *elemNodes2, *elemEdges1, *elemEdges2;
  int fis[nn], fjs[nn], ls[nn];
  fiTimes3 = 0;
  for (i=0; i<=(3*ne-1)/nn+1; i++)
    {
      idx1 = i*nn;
      idx2 = mymin<int>(3*ne,(i+1)*nn);
      ni = idx2-idx1;
      for (j=idx1, idx=0; j<idx2; j++, idx++, fiTimes3++)
        {
          fi = fiTimes3/3+1;
          elemNodes1 = elemNodes+3*(fi-1);
          elemEdges1 = elemEdges+3*(fi-1);
          fj = commEdgeElems2Elems[j];
          elemNodes2 = elemNodes+3*(fj-1);
          elemEdges2 = elemEdges+3*(fj-1);
          for (k=0; k<3; k++)
            {
              l = Find<int>(-elemEdges1[k],elemEdges2,3);
              if (l>0)
                {
                  l = l-1;
                  memcpy(P1,nodes+3*(elemNodes1[(k+1)%3]-1),3*sizeof(double));
                  memcpy(P2,nodes+3*(elemNodes1[k]-1),3*sizeof(double));
                  memcpy(P3,nodes+3*(elemNodes1[(k+2)%3]-1),3*sizeof(double));
                  memcpy(Q1,nodes+3*(elemNodes2[l]-1),3*sizeof(double));
                  memcpy(Q2,nodes+3*(elemNodes2[(l+1)%3]-1),3*sizeof(double));
                  memcpy(Q3,nodes+3*(elemNodes2[(l+2)%3]-1),3*sizeof(double));
                  ls[idx] = l;
                  break;
                }
            }
          for (k=0; k<3; k++)
            {
              us[3*idx+k] = (float) (P2[k]-P1[k]);
              vs[3*idx+k] = (float) (P3[k]-P2[k]);
              ws[3*idx+k] = (float) (Q3[k]-Q2[k]);
            }
          fis[idx] = fiTimes3/3+1;
          fjs[idx] = fj;
        }
      cudaMemcpy(dev_us,us,3*ni*sizeof(float),cudaMemcpyHostToDevice);
      cudaMemcpy(dev_vs,vs,3*ni*sizeof(float),cudaMemcpyHostToDevice);
      cudaMemcpy(dev_ws,ws,3*ni*sizeof(float),cudaMemcpyHostToDevice);
      doubleLayerLaplace3d_commonEdge<<<nblocks,nthreads>>>(dev_us,dev_vs,
                                                            dev_ws,
                                                            quadn,dev_Matrix);
      cudaMemcpy(Matrix,dev_Matrix,3*ni*sizeof(float),cudaMemcpyDeviceToHost);
      for (idx=0; idx<ni; idx++)
        {
          elemNodes2 = elemNodes+3*(fjs[idx]-1);
          for (k=0; k<3; k++)
            result[fis[idx]-1] += ((double)(Matrix[3*idx+k]))
              * vec[elemNodes2[(ls[idx]+k)%3]-1];
        }
    }

  // Assemble common-vertex entries
  fi = 1;
  int fiIdx = 0;
  for (i=0; i<=(commNodesNTotal-1)/nn+1; i++)
    {
      idx1 = i*nn;
      idx2 = mymin<int>(commNodesNTotal,(i+1)*nn);
      ni = idx2-idx1;
      for (j=idx1, idx=0; j<idx2; j++, idx++)
        {
          while (commNodeElems2ElemsSizes[fi-1]==0)
            {
              fi++;
              fiIdx = 0;
            }
          elemNodes1 = elemNodes+3*(fi-1);
          elemEdges1 = elemEdges+3*(fi-1);
          fj = commNodeElems2Elems[commNodeElems2ElemsIdx[fi-1]+fiIdx];
          elemNodes2 = elemNodes+3*(fj-1);
          elemEdges2 = elemEdges+3*(fj-1);
          for (k=0; k<3; k++)
            {
              l = Find<int>(elemNodes1[k],elemNodes2,3);
              if (l>0)
                {
                  l = l-1;
                  memcpy(P1,nodes+3*(elemNodes1[k]-1),3*sizeof(double));
                  memcpy(P2,nodes+3*(elemNodes1[(k+1)%3]-1),3*sizeof(double));
                  memcpy(P3,nodes+3*(elemNodes1[(k+2)%3]-1),3*sizeof(double));
                  memcpy(Q1,nodes+3*(elemNodes2[l]-1),3*sizeof(double));
                  memcpy(Q2,nodes+3*(elemNodes2[(l+1)%3]-1),3*sizeof(double));
                  memcpy(Q3,nodes+3*(elemNodes2[(l+2)%3]-1),3*sizeof(double));
                  ls[idx] = l;
                  break;
                }
            }
          for (k=0; k<3; k++)
            {
              us[3*idx+k] = (float) (P2[k]-P1[k]);
              vs[3*idx+k] = (float) (P3[k]-P2[k]);
              ws[3*idx+k] = (float) (Q2[k]-Q1[k]);
              zs[3*idx+k] = (float) (Q3[k]-Q2[k]);
            }
          fis[idx] = fi;
          fjs[idx] = fj;
          if (fiIdx+1<commNodeElems2ElemsSizes[fi-1])
            fiIdx++;
          else
            {
              fi++;
              fiIdx = 0;
            }
        }
      cudaMemcpy(dev_us,us,3*ni*sizeof(float),cudaMemcpyHostToDevice);
      cudaMemcpy(dev_vs,vs,3*ni*sizeof(float),cudaMemcpyHostToDevice);
      cudaMemcpy(dev_ws,ws,3*ni*sizeof(float),cudaMemcpyHostToDevice);
      cudaMemcpy(dev_zs,zs,3*ni*sizeof(float),cudaMemcpyHostToDevice);
      doubleLayerLaplace3d_commonVertex<<<nblocks,nthreads>>>(dev_us,dev_vs,
                                                              dev_ws,dev_zs,
                                                              quadn,
                                                              dev_Matrix);
      cudaMemcpy(Matrix,dev_Matrix,3*ni*sizeof(float),cudaMemcpyDeviceToHost);
      for (idx=0; idx<ni; idx++)
        {
          elemNodes2 = elemNodes+3*(fjs[idx]-1);          
          for (k=0; k<3; k++)
            result[fis[idx]-1] += ((double)(Matrix[3*idx+k]))
              * vec[elemNodes2[(ls[idx]+k)%3]-1];
        }
    }
  
  // Assemble disjoint-panels entries
  fi = 1; fj = 1;
  long int iidx1, iidx2;
  for (i=0; i<=(disjPanelsNTotal-1)/nn+1; i++)
    {
      iidx1 = i*((long int)nn);
      iidx2 = mymin<long int>(disjPanelsNTotal,(i+1)*((long int)nn));
      ni = ((int)(iidx2-iidx1));
      for (idx=0; idx<ni; idx++)
        {
          while (1)
            {
              // skip identical panels
              if (fi==fj)
                {
                  nextIteration(fi,fj,1,ne);
                  continue;
                }
              // skip common-edge panels
              if (commEdgeElems2Elems[3*(fi-1)]==fj ||
                  commEdgeElems2Elems[3*(fi-1)+1]==fj ||
                  commEdgeElems2Elems[3*(fi-1)+2]==fj)
                {
                  nextIteration(fi,fj,1,ne);
                  continue;
                }
              // skip common-vertex panels
              for (k=0; k<commNodeElems2ElemsSizes[fi-1]; k++)
                if (commNodeElems2Elems[commNodeElems2ElemsIdx[fi-1]+k]==fj)
                  break;
              if (k<commNodeElems2ElemsSizes[fi-1])
                {
                  nextIteration(fi,fj,1,ne);
                  continue;
                }
              break;
            }
          elemNodes1 = elemNodes+3*(fi-1);
          elemEdges1 = elemEdges+3*(fi-1);
          elemNodes2 = elemNodes+3*(fj-1);
          elemEdges2 = elemEdges+3*(fj-1);
          memcpy(P1,nodes+3*(elemNodes1[0]-1),3*sizeof(double));
          memcpy(P2,nodes+3*(elemNodes1[1]-1),3*sizeof(double));
          memcpy(P3,nodes+3*(elemNodes1[2]-1),3*sizeof(double));
          memcpy(Q1,nodes+3*(elemNodes2[0]-1),3*sizeof(double));
          memcpy(Q2,nodes+3*(elemNodes2[1]-1),3*sizeof(double));
          memcpy(Q3,nodes+3*(elemNodes2[2]-1),3*sizeof(double));
          for (k=0; k<3; k++)
            {
              As[3*idx+k] = (float) (P1[k]);
              us[3*idx+k] = (float) (P2[k]-P1[k]);
              vs[3*idx+k] = (float) (P3[k]-P1[k]);
              Bs[3*idx+k] = (float) (Q1[k]);
              ws[3*idx+k] = (float) (Q2[k]-Q1[k]);
              zs[3*idx+k] = (float) (Q3[k]-Q1[k]);
            }
          fis[idx] = fi;
          fjs[idx] = fj;
          nextIteration(fi,fj,1,ne);
        }
      cudaMemcpy(dev_As,As,3*ni*sizeof(float),cudaMemcpyHostToDevice);
      cudaMemcpy(dev_us,us,3*ni*sizeof(float),cudaMemcpyHostToDevice);
      cudaMemcpy(dev_vs,vs,3*ni*sizeof(float),cudaMemcpyHostToDevice);
      cudaMemcpy(dev_Bs,Bs,3*ni*sizeof(float),cudaMemcpyHostToDevice);
      cudaMemcpy(dev_ws,ws,3*ni*sizeof(float),cudaMemcpyHostToDevice);
      cudaMemcpy(dev_zs,zs,3*ni*sizeof(float),cudaMemcpyHostToDevice);
      doubleLayerLaplace3d_disjointPanels<<<nblocks,nthreads>>>
        (dev_As,dev_us,dev_vs,dev_Bs,dev_ws,dev_zs,quadTrin,dev_Matrix);
      cudaMemcpy(Matrix,dev_Matrix,3*ni*sizeof(float),cudaMemcpyDeviceToHost);
      for (idx=0; idx<ni; idx++)
        {
          elemNodes2 = elemNodes+3*(fjs[idx]-1);          
          for (k=0; k<3; k++)
            result[fis[idx]-1] += ((double)(Matrix[3*idx+k]))
              * vec[elemNodes2[k]-1];
        }
      progress = (100*i)/ne;
      std::cout << "\rBEM 3d Laplace GPU-apply 2-layer - "
                << nnodes << " nodes, "
                << ne << " elements: " << progress << "%" << std::flush;
    }
  std::cout << "\rBEM 3d Laplace GPU-apply 2-layer - " << nnodes << " nodes, "
            << ne << " elements: 100%" << std::flush;
  std::cout << std::endl;
}


void applyGPULaplace3dHyperSing (const double *vec, double *result)
{ }

