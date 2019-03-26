#include <cuda_runtime.h>
#include <Rinternals.h>

#include "Rmpi.h"


extern "C" SEXP R_dimrgame_init(SEXP comm_)
{
  int ngpus;
  int rank;
  int id;
  
  MPI_Comm comm = get_mpi_comm_from_Robj(comm_);
  MPI_Comm_rank(comm, &rank);
  
  cudaGetDeviceCount(&ngpus);
  
  id = rank % ngpus;
  cudaSetDevice(id);
  
  return R_NilValue;
}
