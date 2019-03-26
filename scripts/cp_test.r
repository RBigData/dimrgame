suppressMessages(library(float))
suppressMessages(library(dimrgame))

if (comm.size() > 1)
  comm.stop("run with 1 rank")

x = matrix(1:30, 10)
s = fl(x)

ds = shaq(s)

truth = crossprod(s)
test = crossprod_game(ds)
all.equal(test, truth, tolerance=1e-4)

finalize()
