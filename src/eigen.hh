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



static inline void eigen_err_init(eigen_err_t *err)
{
  err->status = CUSOLVER_STATUS_SUCCESS;
  err->code = cudaSuccess;
  err->info = 0;
}



#define EIGEN_CHECK(err) if(!eigen_err_check(err)) return err;

static inline bool eigen_err_check(eigen_err_t err)
{
  if (err.status != CUSOLVER_STATUS_SUCCESS || err.code != cudaSuccess || err.info != 0)
    return true;
  else
    return false;
}



static inline void eigen_err_throw(eigen_err_t err)
{
  if (err.status != CUSOLVER_STATUS_SUCCESS)
  {
    if (err.status == CUSOLVER_STATUS_NOT_INITIALIZED)
      error("cuSolver could not be initialized");
    else if (err.status == CUSOLVER_STATUS_ALLOC_FAILED)
      error("cuSolver could not successfully allocate memory");
    else if (err.status == CUSOLVER_STATUS_INVALID_VALUE)
      error("cuSolver function received invalid input value");
    else if (err.status == CUSOLVER_STATUS_ARCH_MISMATCH)
      error("cuSolver requires feature missing from device arch");
    else if (err.status == CUSOLVER_STATUS_EXECUTION_FAILED)
      error("cuSolver failed to execute");
    else if (err.status == CUSOLVER_STATUS_INTERNAL_ERROR)
      error("cuSolver experienced an internal error");
    else if (err.status == CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED)
      error("cuSolver received unsupported matrix type");
    else
      error("unknown cuSolver error");
  }
  else if (err.code != cudaSuccess)
    error(""); // TODO
  else if (err.info != 0)
    error("syevd returned info=%d", err.info);
}



// NOTE can't template because of extern temp storage
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



static inline cusolverStatus_t cusolver_syevd_buffersize(cusolverDnHandle_t handle,
  cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, float *x, float *values,
  int *lwork)
{
  return cusolverDnSsyevd_bufferSize(handle, jobz, uplo, n, x, n, values, lwork);
}

static inline cusolverStatus_t cusolver_syevd_buffersize(cusolverDnHandle_t handle,
  cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, double *x, double *values,
  int *lwork)
{
  return cusolverDnDsyevd_bufferSize(handle, jobz, uplo, n, x, n, values, lwork);
}



static inline cusolverStatus_t cusolver_syevd(cusolverDnHandle_t handle,
  cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, float *x, float *values,
  float *work, int lwork, int *info_gpu)
{
  return cusolverDnSsyevd(handle, jobz, uplo, n, x, n, values, work, lwork, info_gpu);
}

static inline cusolverStatus_t cusolver_syevd(cusolverDnHandle_t handle,
  cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, double *x, double *values,
  double *work, int lwork, int *info_gpu)
{
  return cusolverDnDsyevd(handle, jobz, uplo, n, x, n, values, work, lwork, info_gpu);
}



template <typename REAL>
static inline void syevd_gpu(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
  cublasFillMode_t uplo, int n, REAL *x, REAL *values, eigen_err_t *err)
{
  int lwork;
  int *info_gpu;
  REAL *work;
  
  err->status = cusolver_syevd_buffersize(handle, jobz, uplo, n, x, values, &lwork);
  cudaMalloc(&work, lwork*sizeof(*work));
  cudaMalloc(&info_gpu, sizeof(*info_gpu));
  
  err->status = cusolver_syevd(handle, jobz, uplo, n, x, values, work, lwork, info_gpu);
  cudaFree(work);
  
  cudaMemcpy(&(err->info), info_gpu, sizeof(*info_gpu), cudaMemcpyDeviceToHost);
  cudaFree(info_gpu);
}



template <typename REAL>
static inline eigen_err_t eigen(int only_values, int n, REAL *const restrict x, REAL *const restrict values)
{
  eigen_err_t err;
  cusolverDnHandle_t handle;
  
  eigen_err_init(&err);
  
  cusolverEigMode_t jobz = only_values ? CUSOLVER_EIG_MODE_NOVECTOR : CUSOLVER_EIG_MODE_VECTOR;
  cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
  
  err.status = cusolverDnCreate(&handle);
  EIGEN_CHECK(err);
  
  syevd_gpu(handle, jobz, uplo, n, x, values, &err);
  EIGEN_CHECK(err);
  
  err.status = cusolverDnDestroy(handle);
  EIGEN_CHECK(err);
  
  revsqrt<<<1, n, n*sizeof(*values)>>>(n, values);
  
  return err;
}


#endif
