#ifndef DIMRGAME_MPI_UTILS_H_
#define DIMRGAME_MPI_UTILS_H_


#define OMPI_SKIP_MPICXX 1
#include <mpi.h>


static inline int allreduce_real(const int len, float *const __restrict__ x, const MPI_Comm comm)
{
  return MPI_Allreduce(MPI_IN_PLACE, x, len, MPI_FLOAT, MPI_SUM, comm);
}

static inline int allreduce_real(const int len, double *const __restrict__ x, const MPI_Comm comm)
{
  return MPI_Allreduce(MPI_IN_PLACE, x, len, MPI_DOUBLE, MPI_SUM, comm);
}


#endif
