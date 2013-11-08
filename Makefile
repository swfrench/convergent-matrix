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

# build products
OBJ = $O/simple_example.o
BIN = $B/simple_example.x

####

all : $(BIN)

$(BIN) : $O $B $(OBJ)
	$(CXX) $(UPCXX_LDFLAGS) $(BLAS_LDFLAGS) \
		$(OBJ) $(UPCXX_LDLIBS) $(BLAS_LDLIBS) -o $@

$O/%.o : %.cpp $I/*.hpp
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
