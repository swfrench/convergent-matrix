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

A working MPI implementation is also required, and your code must be compiled
with `ENABLE_MPIIO_SUPPORT`, if the MPI-IO-based `load()` and `save()` methods
are to be available.

**Note**: In order to use the `load()` and `save()` methods, which use MPI-IO
to read / write distributed matrix data to disk (in hopes of taking advantage
of collective buffering optimizations, etc.), you must first initialize MPI in
your program. Further, if you have compiled `ConvergentMatrix` with support for
a progress thread, you must initialize MPI with thread support at the level of
`MPI_THREAD_FUNNELED` or higher.

## Key feature macros

Enabling production features:
* `-DENABLE_MPIIO_SUPPORT`: enable MPI-IO based `load()` and `save()` methods
  for reading / writing matrix data
* `-DENABLE_PROGRESS_THREAD`: enable progress thread support (note: the progress
  thread must still be turned on / off at run time; see the
  `progress_thread_start` and `progress_thread_stop` methods)
* `-DENABLE_FLUSH_ALLOC_RETRY`: enable bounded exponential backoff for remote
  allocations in `bin::flush` (minimum retry interval, maximum retry interval,
  prolongation factor, and maximum retry count are configurable as well; see
  `include/bin.hpp`)

Enabling testing features:
* `-DENABLE_CONSISTENCY_CHECK`: replicate the full matrix on each participating
  UPC++ process, applying updates to both the local replica and the distributed
  matrix; global reduction across replicas in `ConvergentMatrix::commit` is
  then used as a "ground truth" to assess correctness of distributed matrix
  elements. Only works for (small) matrices that fit in memory.

## References

[1] Y. Zheng, et al., "UPC++: A PGAS Extension for C++," accepted to IPDPS 2014.
