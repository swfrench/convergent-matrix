include ../flags.mk

CXX = $(UPCXXPATH)/bin/upcxx

RUN = $(UPCXXPATH)/bin/upcxx-run

UPCXXFLAGS = -codemode=O3 -network=smp

CXXFLAGS = $(UPCXXFLAGS) -std=c++14 -I..

LDFLAGS = $(UPCXXFLAGS)

O = obj
B = bin

all : bin_test cm_test

$O :
	@mkdir -p $O

$B :
	@mkdir -p $B

$O/%.o : %.cpp ../include/bin.hpp ../include/convergent_matrix.hpp | $O
	$(CXX) $(CXXFLAGS) -c $< -o $@

$B/bin_test : $O/bin_test.o | $B
	$(CXX) $(LDFLAGS) $< -o $@

.PHONY : bin_test
bin_test : $B/bin_test
	@$(RUN) -n 4 $B/bin_test && echo OK

$B/cm_test : $O/cm_test.o | $B
	$(CXX) $(LDFLAGS) $< -o $@

.PHONY : cm_test
cm_test : $B/cm_test
	@$(RUN) -n 4 $B/cm_test && echo OK

.PHONY : clean
clean :
	rm -rf $O

.PHONY : distclean
distclean :
	rm -rf $O $B

.PHONY : fmt
fmt :
	clang-format -i bin_test.cpp
