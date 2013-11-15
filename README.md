A "convergent" distributed dense matrix data structure
------------------------------------------------------

**Note**: the master branch will not build with the current master branch of
`upcxx`, as it does not have the `drain()` function. The latter is implemented
in the `drain` branch for the time being.

A preliminary implementation of a distributed dense-matrix data structure
supporting asynchronous updates.

This abstraction is "convergent" in that updates are cached locally and applied
to the distributed matrix asynchronously. The global distributed matrix is
guaranteed to converge to its final state some time after all updates have been
applied and we call `ConvergentMatrix::freeze()`.
