#ifndef DIMRGAME_CROSSPROD_H_
#define DIMRGAME_CROSSPROD_H_


#include "blas.hh"
#include "cu_utils.hh"
#include "mpi_utils.hpp"
#include "types.hpp"


template <typename REAL>
static inline void crossprod(const shaq_t<REAL> *const restrict dx, REAL *const restrict cp)
{
  const int m = NROWS_LOCAL(dx);
  const int n = NCOLS_LOCAL(dx);
  MPI_Comm comm = COMM(dx);
  
  int nb = get_num_blocks(m);
  
  cublasHandle_t handle;
  cublasStatus_t st = cublasCreate_v2(&handle);
  if (st != CUBLAS_STATUS_SUCCESS)
    error("todo");
  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
  
  REAL *x_gpu;
  REAL *cp_gpu;
  cudaMalloc(&x_gpu, m*n*sizeof(*x_gpu));
  cudaMalloc(&cp_gpu, n*n*sizeof(*cp_gpu));
  cudaMemcpy(x_gpu, DATA(dx), m*n*sizeof(REAL), cudaMemcpyHostToDevice);
  
  crossprod_mat(handle, m, n, x_gpu, cp_gpu);
  cublasDestroy_v2(handle);
  cudaFree(x_gpu);
  
  cudaMemcpy(cp, cp_gpu, n*n*sizeof(REAL), cudaMemcpyDeviceToHost);
  cudaFree(cp_gpu);
  
  int check = allreduce_real(n, cp, comm);
  if (check != MPI_SUCCESS)
    R_err_mpi(check, comm);
}


#endif
