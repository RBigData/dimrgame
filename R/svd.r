#' @useDynLib dimrgame R_svd
#' @export
svd_game = function(x)
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
  
  #FIXME
  retu = FALSE
  retv = FALSE
  
  ret = .Call(R_svd, data, retu, retv, comm_ptr)
  
  if (is_float)
    ret$d = float32(ret$d)
  
  ret
}
