#ifndef DIMRGAME_CU_UTILS_H_
#define DIMRGAME_CU_UTILS_H_


#include <cuda_runtime.h>

#define TPB 512

#define CUFREE(x) {if(x)cudaFree(x);}
#define PRINT_CUDA_ERROR() printf("%s\n", cudaGetErrorString(cudaGetLastError()));

static inline int get_num_blocks(const int m)
{
  int nb = m / TPB;
  if (m % TPB)
    nb++;
  
  return nb;
}


#endif
