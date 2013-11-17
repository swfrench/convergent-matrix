# local path to upcxx install
UPCXX_PATH = $(HOME)/Code/Dev/upcxx/upcxx_install

# C++ compiler
CXX = clang++

# base compiler opts
CXXFLAGS = -O3 -Wall -DNOCHECK

# blas (uses fortran interface internally, so no CXXFLAGS)
BLAS_LDFLAGS =
BLAS_LDLIBS = -framework Accelerate
