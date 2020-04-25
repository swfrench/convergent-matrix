# Install prefix for upcxx
UPCXXPATH = 

# Default upcxx compiler options appropriate for local testing
UPCXXFLAGS = -codemode=O3 -network=smp

# CXXFLAGS appropriate for use with ConvergentMatrix
CXXFLAGS = $(UPCXXFLAGS) -std=c++14

# C++ compiler
CXX = $(UPCXXPATH)/bin/upcxx

# Linker flags to support BLAS dependency in LocalMatrix (uses fortran
# interface internally, so no CXXFLAGS are required).
BLAS_LDFLAGS =
BLAS_LDLIBS = -lblas
