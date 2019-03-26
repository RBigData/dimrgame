/* Automatically generated. Do not edit by hand. */
  
#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <stdlib.h>

extern SEXP R_crossprod(SEXP m, SEXP data, SEXP comm_);
extern SEXP R_dimrgame_init(SEXP comm_);
extern SEXP R_svd(SEXP data, SEXP retu_, SEXP retv_, SEXP comm_);

static const R_CallMethodDef CallEntries[] = {
  {"R_crossprod", (DL_FUNC) &R_crossprod, 3},
  {"R_dimrgame_init", (DL_FUNC) &R_dimrgame_init, 1},
  {"R_svd", (DL_FUNC) &R_svd, 4},
  {NULL, NULL, 0}
};

void R_init_dimrgame(DllInfo *dll)
{
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}
