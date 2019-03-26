#ifndef DIMRGAME_RMPI_H_
#define DIMRGAME_RMPI_H_


#ifdef __cplusplus
#define OMPI_SKIP_MPICXX 1
#endif
#include <mpi.h>
#include <Rinternals.h>


#define MPI_CHECK(comm, check) if (check != MPI_SUCCESS) R_err_mpi(check, comm);

static inline void R_err_mpi(int check, const MPI_Comm comm)
{
  int rank;
  MPI_Comm_rank(comm, &rank);
  if (rank == 0)
    error("MPI_Allreduce returned error code %d\n", check);
  else
    error(""); // FIXME
}



static inline void R_err_malloc(const MPI_Comm comm)
{
  int rank;
  MPI_Comm_rank(comm, &rank);
  if (rank == 0)
    error("Out of memory\n");
  else
    error(""); // FIXME
}



static inline MPI_Comm get_mpi_comm_from_Robj(SEXP comm_)
{
  MPI_Comm *comm = (MPI_Comm*) R_ExternalPtrAddr(comm_);
  return *comm;
}


#endif
