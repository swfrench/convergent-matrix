A distributed dense matrix abstraction
--------------------------------------

`ConvergentMatrix` is a dense matrix abstraction for distributed-memory systems,
supporting matrix assembly via asynchronous _commutative_ update operations.
The abstraction is based on UPC++, an experimental PGAS extension to the C++
language [1], which in turn uses GASNet [2] for high-performance one-sided
remote memory access and messaging.

The abstraction is "convergent" in that updates are cached locally and applied
to the distributed matrix asynchronously (potentially out of order). Concurrent
updates on the same matrix elements are scheduled to execute in isolation from
one another.
The global distributed matrix converges to its final state after all updates
have been applied and the user calls `ConvergentMatrix::commit()`.

[1] Y. Zheng, et al., "UPC++: A PGAS Extension for C++," accepted to IPDPS 2014.

[2] http://gasnet.lbl.gov
