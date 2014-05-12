A PGAS distributed matrix abstraction
-------------------------------------

`ConvergentMatrix` is a dense matrix abstraction for distributed-memory systems,
supporting matrix assembly via asynchronous _commutative_ update operations.

The abstraction is based on [UPC++](https://bitbucket.org/upcxx/upcxx "UPC++"),
a new PGAS extension to the C++ language [1], which in turn uses
[GASNet](http://gasnet.lbl.gov "GASNet") for high-performance one-sided
remote memory access and active messaging.

Updates to matrix elements are cached locally and applied to the distributed
matrix asynchronously, potentially out of order. Concurrent updates on the same
elements are scheduled to execute in isolation from one another.
The global distributed matrix "converges" to its final state after all updates
have been applied and the user calls `ConvergentMatrix::commit()`.

[1] Y. Zheng, et al., "UPC++: A PGAS Extension for C++," accepted to IPDPS 2014.
