/*file name [matrixmul_gold.cpp]
Synopsis [This file defines the gold-version matrix-matrix
multiplication.]
Description []
*******************************************************/
#include <stdio.h>
#include "matrixmul.h"
/*=========================================*/
/* */
/* Synopsis [Sequential/Gold version of matrix-matrix
multiplication.] */
/* */
/* Description [This function computes multiplication
of two matrix M and N, */
/* and stores the output to P.] */
/* */
/*=========================================*/
void computeGold(float *P, const float *M, const float *N, int Mh, int Mw, int Nw)
{
  int i, j, k;
  float sum, a, b;
  for (i = 0; i < Mh; i++)
  {
    for (j = 0; j < Nw; j++)
    {
      sum = 0;
      for (k = 0; k < Mw; k++)
      {
        a = M[i * Mw + k];
        b = N[k * Nw + j];
        //printf ("A[%d] * B[%d]\n", i * Mw + k, k * Nw + j);
        sum += a * b;
      }
      P[i * Nw + j] = (float)sum;
    }
  }
}
