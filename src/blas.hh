#ifndef DIMRGAME_BLAS_H_
#define DIMRGAME_BLAS_H_


#include <cublas_v2.h>
#include "restrict.h"

static inline void crossprod_mat(cublasHandle_t handle, const int m, const int n, const float *const restrict x, float *const restrict cp)
{
  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m, &alpha, x, m, x, m, &beta, cp, n);
}

static inline void crossprod_mat(cublasHandle_t handle, const int m, const int n, const double *const restrict x, double *const restrict cp)
{
  const double alpha = 1.0;
  const double beta = 0.0;
  cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m, &alpha, x, m, x, m, &beta, cp, n);
}


#endif
