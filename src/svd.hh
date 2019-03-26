#ifndef DIMRGAME_SVD_H_
#define DIMRGAME_SVD_H_


#include "crossprod.hh"
#include "cu_utils.hh"
#include "eigen.hh"
#include "restrict.h"
#include "types.hpp"


template <typename REAL>
struct svd_t
{
  REAL *restrict d;
  REAL *restrict u;
  REAL *restrict v;
};



template <typename REAL>
static inline void svd(const shaq_t<REAL> *const restrict dx, svd_t<REAL> *const restrict s)
{
  const int n = NCOLS_LOCAL(dx);
  
  REAL *cp = (REAL*) malloc(n*n*sizeof(*cp));
  crossprod(dx, cp);
  
  REAL *cp_gpu;
  REAL *d_gpu;
  cudaMalloc(&cp_gpu, n*n*sizeof(*cp_gpu));
  cudaMalloc(&d_gpu, n*sizeof(*d_gpu));
  cudaMemcpy(cp_gpu, cp, n*n*sizeof(*cp_gpu), cudaMemcpyHostToDevice);
  free(cp);
  
  eigen(1, n, cp_gpu, d_gpu);
  
  cudaFree(cp_gpu);
  cudaMemcpy(s->d, d_gpu, n*sizeof(*d_gpu), cudaMemcpyDeviceToHost);
  cudaFree(d_gpu);
}


#endif
