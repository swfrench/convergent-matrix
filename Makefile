# pull in config
include flags.mk

# pull in upcxx flags
include $(UPCXX_PATH)/include/upcxx.mak

# local include dir
I = include

# compilation
CXXFLAGS += $(UPCXX_CXXFLAGS) -I$I

# dirs
O = obj
B = bin
D = doc
T = tests

# build products
OBJ = $O/simple.o
BIN = $B/simple.x
TOBJ = $O/simple_test.o
TBIN = $B/simple_test.x

####

# default target: the simple example
all : $(BIN)

$(BIN) : $O $B $(OBJ)
	$(CXX) $(UPCXX_LDFLAGS) $(BLAS_LDFLAGS) \
		$(OBJ) $(UPCXX_LDLIBS) $(BLAS_LDLIBS) -o $@

$(TBIN) : $O $B $(TOBJ)
	$(CXX) $(UPCXX_LDFLAGS) $(BLAS_LDFLAGS) \
		$(TOBJ) $(UPCXX_LDLIBS) $(BLAS_LDLIBS) -o $@

$O/%.o : example/%.cpp $I/*.hpp
	$(CXX) $(CXXFLAGS) -DENABLE_CONSISTENCY_CHECK -c $< -o $@

$O/%.o : $T/%.cpp $I/*.hpp $T/*.hpp
	$(CXX) $(CXXFLAGS) -I$T \
		-DUSE_MPI_WTIME -DENABLE_CONSISTENCY_CHECK -c $< -o $@

## compile the test; display run hint
.PHONY : test
test : $(TBIN)
	@echo
	@echo "Now, $(TBIN) must be run with a sufficient number of threads"
	@echo "to accommodate the following process grid:"
	@grep ^#define tests/simple_test_setup.hpp | grep NP[RC]
	@echo

## docs
.PHONY : $D
$D :
	doxygen doxygen.conf

## cleanup
.PHONY : clean
clean :
	rm -rf $O
.PHONY : distclean
distclean :
	rm -rf $O $B $D

## paths for build products
$O :
	mkdir -p $O
$B :
	mkdir -p $B
