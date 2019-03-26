#ifndef DIMRGAME_TYPES_H_
#define DIMRGAME_TYPES_H_


#include <cstdint>

#define MPI_LEN_T MPI_UINT64_T
#define MPI_LEN_LOCAL_T MPI_INT

typedef uint64_t len_t;
typedef int len_local_t;

#define NROWS(x) (x->nrows)
#define NCOLS(x) (x->ncols)
#define NROWS_LOCAL(x) (x->nrows_local)
#define NCOLS_LOCAL(x) (x->ncols)
#define DATA(x) (x->data)
#define COMM(x) (x->comm)

#define NOT_REFERENCED 0

template <typename REAL>
struct shaq_t
{
  len_t nrows;
  len_local_t ncols;
  len_local_t nrows_local;
  REAL *restrict data;
  MPI_Comm comm;
};


#endif
