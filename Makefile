# local path to upcxx install
UPCXX_PATH = $(HOME)/Code/Dev/upcxx/upcxx_install

include $(UPCXX_PATH)/include/upcxx.mak

I	= include

CXX	= clang++
CXXFLAGS= -O0 $(UPCXX_CXXFLAGS) -DDEBUGMSGS -I$I

O	= obj
B	= bin

OBJ	= $O/test.o
BIN	= $B/test.x

all : $(BIN)

$(BIN) : $O $B $(OBJ)
	$(CXX) $(UPCXX_LDFLAGS) $(OBJ) $(UPCXX_LDLIBS) -o $@

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
