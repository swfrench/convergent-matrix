
# local path to upcxx install
UPCXX_PATH = $(HOME)/Code/Dev/upcxx/upcxx_install

# pull in upcxx flags
include $(UPCXX_PATH)/include/upcxx.mak

# local include dir
I	= include

# compilation
CXX	= clang++
CXXFLAGS= -O0 $(UPCXX_CXXFLAGS) -DDEBUGMSGS -I$I

# blas (uses fortran interface internally, so no CXXFLAGS)
BLAS_LDFLAGS=
BLAS_LDLIBS = -framework Accelerate

# dirs
O	= obj
B	= bin

# build products
OBJ	= $O/test.o
BIN	= $B/test.x

####

all : $(BIN)

$(BIN) : $O $B $(OBJ)
	$(CXX) $(UPCXX_LDFLAGS) $(BLAS_LDFLAGS) \
		$(OBJ) $(UPCXX_LDLIBS) $(BLAS_LDLIBS) -o $@

$O/%.o : %.cpp $I/convergent_matrix.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$O :
	mkdir -p $O

$B :
	mkdir -p $B

.PHONY : clean
clean :
	rm -rf $O

.PHONY : distclean
distclean :
	rm -rf $O $B
