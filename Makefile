# pull in config
include flags.mk

# pull in upcxx flags
include $(UPCXX_PATH)/include/upcxx.mak

# local include dir
I = include

# compilation
CXXFLAGS += $(UPCXX_CXXFLAGS) -I$I -I.

# dirs
O = obj
B = bin
D = doc

# build products
OBJ = $O/run_ext_test.o
BIN = $B/run_ext_test.x

####

all : $(BIN)

$(BIN) : $O $B $(OBJ)
	$(CXX) $(UPCXX_LDFLAGS) $(BLAS_LDFLAGS) \
		$(OBJ) $(UPCXX_LDLIBS) $(BLAS_LDLIBS) -o $@

$O/%.o : example/%.cpp $I/*.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$O :
	mkdir -p $O

$B :
	mkdir -p $B

.PHONY : $D
$D :
	doxygen doxygen.conf

.PHONY : clean
clean :
	rm -rf $O

.PHONY : distclean
distclean :
	rm -rf $O $B $D
