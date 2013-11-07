A "convergent" global dense matrix data structure
-------------------------------------------------

A preliminary implementation of a distributed dense-matrix data structure
supporting asynchronous updates.

This abstraction is "convergent" in that updates are cached locally and applied
to the distributed matrix asynchronously. The global distributed matrix is
guaranteed to converge to its final state some time after all updates have been
applied and we call `ConvergentMatrix::freeze()`.
