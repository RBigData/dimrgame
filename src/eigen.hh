#ifndef DIMRGAME_EIGEN_H_
#define DIMRGAME_EIGEN_H_


#include <cusolverDn.h>
#include <cmath>

#include "crossprod.hh"
#include "cu_utils.hh"
#include "restrict.h"
#include "types.hpp"


typedef struct eigen_err_t
{
  cusolverStatus_t status;
  cudaError_t code;
  int info;
} eigen_err_t;


#define EIGEN_CHECK_STATUS(status) (status == CUSOLVER_STATUS_SUCCESS)
#define EIGEN_CHECK_CUERR(code) (code == cudaSuccess)
#define EIGEN_CHECK_INFO(info) (info == 0)



__global__ static void revsqrt(const int n, float *x)
{
  extern __shared__ float sf[];
  
  int t = threadIdx.x;
  int tr = n - t - 1;
  
  sf[t] = x[t];
  
  __syncthreads();
  if (sf[tr] < 0)
    x[t] = 0.0f;
  else
    x[t] = sqrt(sf[tr]);
}

__global__ static void revsqrt(const int n, double *x)
{
  extern __shared__ double sd[];
  
  int t = threadIdx.x;
  int tr = n - t - 1;
  
  sd[t] = x[t];
  
  __syncthreads();
  if (sd[tr] < 0)
    x[t] = 0.0;
  else
    x[t] = sqrt(sd[tr]);
}



static inline void cu_syevd(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
  cublasFillMode_t uplo, int n, float *x, float *values, eigen_err_t *err)
{
  int lwork;
  int *info_gpu;
  float *work;
  
  err->status = cusolverDnSsyevd_bufferSize(handle, jobz, uplo, n, x, n, values, &lwork);
  cudaMalloc(&work, lwork*sizeof(*work));
  cudaMalloc(&info_gpu, sizeof(*info_gpu));
  
  err->status = cusolverDnSsyevd(handle, jobz, uplo, n, x, n, values, work, lwork, info_gpu);
  cudaMemcpy(&(err->info), info_gpu, sizeof(*info_gpu), cudaMemcpyDeviceToHost);
  cudaFree(work);
  cudaFree(info_gpu);
}

static inline void cu_syevd(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
  cublasFillMode_t uplo, int n, double *x, double *values, eigen_err_t *err)
{
  int lwork;
  int *info_gpu;
  double *work;
  
  err->status = cusolverDnDsyevd_bufferSize(handle, jobz, uplo, n, x, n, values, &lwork);
  cudaMalloc(&work, lwork*sizeof(*work));
  cudaMalloc(&info_gpu, sizeof(*info_gpu));
  
  err->status = cusolverDnDsyevd(handle, jobz, uplo, n, x, n, values, work, lwork, info_gpu);
  cudaMemcpy(&(err->info), info_gpu, sizeof(*info_gpu), cudaMemcpyDeviceToHost);
  cudaFree(work);
  cudaFree(info_gpu);
}



template <typename REAL>
static inline eigen_err_t eigen(int only_values, int n, REAL *const restrict x, REAL *const restrict values)
{
  eigen_err_t err;
  
  cusolverDnHandle_t handle;
  
  cusolverEigMode_t jobz = only_values ? CUSOLVER_EIG_MODE_NOVECTOR : CUSOLVER_EIG_MODE_VECTOR;
  cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
  
  err.status = cusolverDnCreate(&handle);
  cu_syevd(handle, jobz, uplo, n, x, values, &err);
  err.status = cusolverDnDestroy(handle);
  
  revsqrt<<<1, n, n*sizeof(*values)>>>(n, values);
  
  return err;
}


#endif
