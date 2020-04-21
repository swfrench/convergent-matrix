include ../flags.mk

RUN = $(UPCXXPATH)/bin/upcxx-run

CXXFLAGS += -I..

# Optional: Include MPI-IO support for matrix save / load in cm_test. Requires
# either the use of a programming environment that supports upcxx and MPI side
# by side or (e.g. for local testing) selection of the GASNet mpi-conduit.
CXXFLAGS += -DENABLE_MPIIO_SUPPORT

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
