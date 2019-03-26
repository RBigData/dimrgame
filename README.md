# dimrgame

* **Version:** 0.1-0
* **URL**: https://github.com/RBigData/dimrgame
* **License:** [BSD 2-Clause](http://opensource.org/licenses/BSD-2-Clause)
* **Author:** Drew Schmidt

🚨 Highly experimental 🚨



## Installation

The development version is maintained on GitHub:

```r
remotes::install_github("RBigData/dimrgame")
```

You will need to have an installation of CUDA to build the package. You can download CUDA from the [nvidia website](https://developer.nvidia.com/cuda-downloads). You will also need the development version of the pbdMPI and kazaam packages (and optionally the curand R package):

```r
remotes::install_github("wrathematics/pbdMPI")
remotes::install_github("rbigdata/kazaam")
remotes::install_github("wrathematics/curand")
```

There is a reference cpu version of the package that you can build. However, this is not supported or recommended; please just use kazaam. But if you insist, you can install it via

```r
remotes::install_github("rbigdata/dimrgame", configure.args="--with-backend=CPU")
```
