#include <float/float32.h>
#include <R.h>
#include <Rinternals.h>

#include "common.h"
#include "Rmpi.h"

#include "mpi_utils.hpp"
#include "svd.hh"
#include "types.hpp"


extern "C" SEXP R_svd(SEXP data, SEXP retu_, SEXP retv_, SEXP comm_)
{
  SEXP ret, ret_names;
  SEXP ret_d;
  MPI_Comm comm = get_mpi_comm_from_Robj(comm_);
  
  const int m_local = nrows(data);
  const int n = ncols(data);
  
  
  if (TYPEOF(data) == REALSXP)
  {
    shaq_t<double> x;
    x.nrows = NOT_REFERENCED;
    x.ncols = n;
    x.nrows_local = m_local;
    x.data = REAL(data);
    x.comm = comm;
    
    PROTECT(ret_d = allocVector(REALSXP, n));
    
    svd_t<double> s;
    s.d = REAL(ret_d);
    
    svd(&x, &s);
  }
  else if (TYPEOF(data) == INTSXP)
  {
    shaq_t<float> x;
    x.nrows = NOT_REFERENCED;
    x.ncols = n;
    x.nrows_local = m_local;
    x.data = FLOAT(data);
    x.comm = comm;
    
    PROTECT(ret_d = allocVector(INTSXP, n));
    
    svd_t<float> s;
    s.d = FLOAT(ret_d);
    
    svd(&x, &s);
  }
  else
    error("this should be impossible; please contact the developers\n");
  
  
#define NRET 1
  PROTECT(ret = allocVector(VECSXP, NRET));
  PROTECT(ret_names = allocVector(STRSXP, NRET));
  
  SET_VECTOR_ELT(ret, 0, ret_d);
  SET_STRING_ELT(ret_names, 0, mkChar("d"));
  setAttrib(ret, R_NamesSymbol, ret_names);
  
  UNPROTECT(NRET+2);
#undef NRET
  return ret;
}
