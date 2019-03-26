// Modified from the coop package. Copyright (c) 2016-2017 Drew Schmidt
#include <float/float32.h>
#include <R.h>
#include <Rinternals.h>

#include "common.h"
#include "Rmpi.h"

#include "crossprod.hh"
#include "mpi_utils.hpp"
#include "types.hpp"


extern "C" SEXP R_crossprod(SEXP m, SEXP data, SEXP comm_)
{
  SEXP cp;
  MPI_Comm comm = get_mpi_comm_from_Robj(comm_);
  
  const int m_local = nrows(data);
  const int n = ncols(data);
  
  if (TYPEOF(data) == REALSXP)
  {
    PROTECT(cp = allocMatrix(REALSXP, n, n));
    
    shaq_t<double> x;
    x.nrows = (len_t) REAL(m)[0];
    x.ncols = n;
    x.nrows_local = m_local;
    x.data = REAL(data);
    x.comm = comm;
    
    crossprod(&x, REAL(cp));
  }
  else if (TYPEOF(data) == INTSXP)
  {
    PROTECT(cp = allocMatrix(INTSXP, n, n));
    
    shaq_t<float> x;
    x.nrows = (len_t) REAL(m)[0];
    x.ncols = n;
    x.nrows_local = m_local;
    x.data = FLOAT(data);
    x.comm = comm;
    
    crossprod(&x, FLOAT(cp));
  }
  else
    error("this should be impossible; please contact the developers\n");
  
  
  UNPROTECT(1);
  return cp;
}
