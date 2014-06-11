# A PGAS distributed matrix abstraction

`ConvergentMatrix` is a dense matrix abstraction for distributed-memory HPC
platforms, supporting matrix assembly via asynchronous commutative update
operations.

## Background

The `ConvergentMatrix` abstraction is based on
[UPC++](https://bitbucket.org/upcxx/upcxx "UPC++"), a new PGAS extension to the
C++ language [1].  Under the hood, UPC++ relies on
[GASNet](http://gasnet.lbl.gov "GASNet") for high-performance one-sided remote
memory access and active messaging (AM).

Updates to matrix elements are cached locally and applied to the distributed
matrix asynchronously, potentially out of order. Concurrent updates on the same
elements are scheduled to execute in isolation from one another.  The global
distributed matrix "converges" to its final state after all updates have been
applied and the user calls `ConvergentMatrix::commit()`.

Our solution combines high-performance remote DMA (on supported interconnects)
for bulk transfer of elemental update data, with AM-based asynchronous remote
tasks for update application logic.
This approach allows us to achieve significantly higher update throughput and
enhanced locality over analogous solutions based on MPI-3 RMA operations (e.g.
`MPI_Accumulate`).

## Dependencies

In addition to a working C++ compiler, both UPC++ and GASNet must be installed
in order to use `ConvergentMatrix`.
See [Installing UPC++](https://bitbucket.org/upcxx/upcxx/wiki/Installing%20UPC++
"Installing UPC++") for details.

**Note**: If you are using a recent version of the GNU compilers, you may need
to compile GASNet with the UDP network conduit enabled (regardless of whether
you intend to use it) to ensure that C++ headers are properly generated.
This can be achieved simply by configuring with `--enable-udp`.

## References

[1] Y. Zheng, et al., "UPC++: A PGAS Extension for C++," accepted to IPDPS 2014.
