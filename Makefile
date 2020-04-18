# ConvergentMatrix is entirely header-based, so there are no build artifacts
# defined at this level. See supported targets below.

D = doc
T = test

# Default target: run local tests
all : tests

.PHONY : tests
tests :
	$(MAKE) -C $T

.PHONY : $D
$D :
	doxygen doxygen.conf

.PHONY : clean
clean :
	$(MAKE) -C $T clean

.PHONY : distclean
distclean :
	$(MAKE) -C $T distclean
	rm -rf $D
