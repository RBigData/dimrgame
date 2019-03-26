suppressMessages(library(float))
suppressMessages(library(dimrgame))

if (comm.size() > 1)
  comm.stop("run with 1 rank")

x = matrix(1:30, 10)
# x = matrix(rnorm(30), 10)
s = fl(x)

ds = shaq(s)

truth = svd(s)$d
test = svd_game(ds)$d
truth
test

finalize()
