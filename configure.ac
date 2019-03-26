AC_INIT(DESCRIPTION)
AC_CONFIG_MACRO_DIR([tools/])

: ${R_HOME=`R RHOME`}
if test -z "${R_HOME}"; then
  echo "could not determine R_HOME"
  exit 1
fi
CC=`"${R_HOME}/bin/R" CMD config CC`
R_CPPFLAGS=`"${R_HOME}/bin/R" CMD config --cppflags`

AC_PROG_CC_C99
BACKEND="GPU"
AX_CHECK_CUDA()

SPMD_CPPFLAGS=`"${R_HOME}/bin/Rscript" -e "source('./R/conf/get_conf.r');get.conf('PKG_CPPFLAGS','"${R_ARCH}"')"`
SPMD_LDFLAGS=`"${R_HOME}/bin/Rscript" -e "source('./R/conf/get_conf.r');get.conf('PKG_LIBS','"${R_ARCH}"')"`


# Report
echo " "
echo "***************** Results of dimrgame package configure *****************"
echo " "
echo "* R Report:"
echo "    >> R_CPPFLAGS=${R_CPPFLAGS}"
echo "* Backend:"
echo "    >> BACKEND=${BACKEND}"
echo "* CUDA Report:"
  echo "    >> NVCC=${NVCC}"
  echo "    >> CUDA_CFLAGS=${CUDA_CFLAGS}"
  echo "    >> CUDA_LDFLAGS=${CUDA_LDFLAGS}"
echo "* MPI report:"
echo "    >> SPMD_CPPFLAGS = ${SPMD_CPPFLAGS}"
echo "    >> SPMD_LDFLAGS = ${SPMD_LDFLAGS}"
echo "*************************************************************************"
echo " "



dnl Start to substitute templates
AC_SUBST(NVCC)
AC_SUBST(CUDA_CFLAGS)
AC_SUBST(CUDA_LDFLAGS)
AC_SUBST(R_CPPFLAGS)

AC_SUBST(BACKEND)

AC_SUBST(SPMD_CPPFLAGS)
AC_SUBST(SPMD_LDFLAGS)

AC_OUTPUT(src/Makevars)
