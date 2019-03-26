#' @useDynLib dimrgame R_dimrgame_init
.onLoad = function(libname, pkgname)
{
  s = search()
  if ("package:clustrgame" %in% s || "package:glmrgame" %in% s)
    return(invisible())
  
  comm_ptr = pbdMPI::get.mpi.comm.ptr(.pbd_env$SPMD.CT$comm)
  .Call(R_dimrgame_init, comm_ptr)
  
  invisible()
}
