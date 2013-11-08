# local path to upcxx install
UPCXX_PATH = $(HOME)/upcxx/edison/upcxx_install

# C++ compiler
CXX = CC

# base compiler opts
CXXFLAGS = -O3

# blas (uses fortran interface internally, so no CXXFLAGS)
BLAS_LDFLAGS =
BLAS_LDLIBS =
