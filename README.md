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

You will need to have an installation of CUDA to build the package. You can download CUDA from the [nvidia website](https://developer.nvidia.com/cuda-downloads). You will also need the development versions float, pbdMPI, and kazaam (and optionally the curand R package which is used in some scripts):

```r
remotes::install_github("wrathematics/float")
remotes::install_github("snoweye/pbdMPI")
remotes::install_github("rbigdata/kazaam")
remotes::install_github("wrathematics/curand")
```

Unlike glmrgame and clustrgame, there is no reference CPU version of the code. If you don't have a gpu, just use kazaam instead.
