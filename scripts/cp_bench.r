suppressMessages(library(float))
suppressMessages(library(curand))
suppressMessages(library(dimrgame))

.pbd_env$SPMD.CT$print.quiet = TRUE
comm.cat("----------- crossprod -----------\n")

m = 2000000
n = 250

if (comm.rank() == 0){
  t_gen = comm.timer(x <- matrix(curand::rnorm(m*n), m, n))
  cat("data gen (dbl): ", t_gen[3], "\n")
} else {
  x = NULL
}

dx = expand(x)
ds = dx
ds@Data = fl(DATA(ds))


t_cpu = comm.timer(crossprod(dx))
comm.cat("kazaam (dbl):   ", t_cpu[3], "\n")
t_cpu = comm.timer(crossprod(ds))
comm.cat("kazaam (flt):   ", t_cpu[3], "\n")

t_gpu = comm.timer(crossprod_game(dx))
comm.cat("dimrgame (dbl): ", t_gpu[3], "\n")
t_gpu = comm.timer(crossprod_game(ds))
comm.cat("dimrgame (flt): ", t_gpu[3], "\n")


finalize()
