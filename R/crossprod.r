#' crossprod_game
#' 
#' Crossproduct on a gpu.
#' 
#' @section Communication:
#' The operation consists of a local crossproduct, followed by an
#' \code{allreduce()} call, quadratic on the number of columns.
#' 
#' For \code{crossprod()}, if the matrix distribution is poorly balanced
#' (specifically, if any rank has fewer rows than columns), then an inefficient
#' method is used. Similarly for \code{tcrossprod()} if the number of local rows
#' is greater than the number of local columns.
#' 
#' @param x
#' A shaq.
#' 
#' @return 
#' A regular matrix.
#' 
#' @useDynLib dimrgame R_crossprod
#' @export
crossprod_game = function(x)
{
  comm_ptr = pbdMPI::get.mpi.comm.ptr(.pbd_env$SPMD.CT$comm)
  is_float = is.float(DATA(x))
  
  if (is_float)
    data = DATA(x)@Data
  else
  {
    data = DATA(x)
    if (!is.double(data))
      storage.mode(data) = "double"
  }
  
  m = as.double(nrow(x))
  ret = .Call(R_crossprod, m, data, comm_ptr)
  
  if (is_float)
    ret = float32(ret)
  
  ret
}
