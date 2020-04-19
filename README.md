# A PGAS distributed matrix abstraction

`ConvergentMatrix` is a dense matrix abstraction for distributed-memory HPC
platforms, supporting matrix assembly via asynchronous commutative one-sided
update operations.

## Background

`ConvergentMatrix` is based on
[UPC++](https://bitbucket.org/berkeleylab/upcxx/wiki/Home "UPC++"), a
communication library for C++ supporting the Partitioned Global Address Space
(PGAS) model. Under the hood, UPC++ relies on [GASNet-EX](http://gasnet.lbl.gov
"GASNet-EX") for high-performance one-sided remote memory access and
active-messaging (AM) based remote procedure calls.

The original version of `ConvergentMatrix` based on UPC++ v0.1 was described in
[1] and used in the development of the SEMUCB-WM1 tomographic model [2,3]. The
former contains details on why UPC++ was adopted over an analogous solution
based on MPI-3 RMA operations, as well as comparison benchmarks.

Since then, `ConvergentMatrix` has been ported to the UPC++ v1.0 spec and
adopted newly supported idioms (e.g. replacing remote allocation + one-sided
copy + asynchronous remote task invocation with simpler serialization + RPC).

## Properties

`ConvergentMatrix` accepts additive updates to distributed matrix elements,
which are buffered locally and applied asynchronously, potentially out of
order (see `ConvergentMatrix::update`).

The distributed matrix "converges" to its final state after all matrix updates
have been requested and `ConvergentMatrix::commit` called by all participating
processes (a collective operation). After `commit` returns, the
PBLAS-compatible (e.g. for use with ScaLAPACK) portion of the distributed
matrix local to each process is fully up to date.

**Progress**: It is assumed that `ConvergentMatrix` instances are only
manipulated by the thread holding the master persona on a participating
process. This ensures that assumptions surrounding quiescence in methods such
as `commit` hold (i.e. entering operations that ensure user-level progress
therein will execute remotely injected updates). See the UPC++ Programming
Guide or Specification for more details.

**Thread safety**: `ConvergentMatrix` is not thread safe (see also notes on
Progress above).

## Dependencies

`ConvergentMatrix` requires a C++ compiler supporting the C++11 standard or
higher, as well as a working UPC++ installation (see
[instructions](https://bitbucket.org/berkeleylab/upcxx/wiki/INSTALL)). At a
minimum (i.e. to run the local test suite), you will need support for the SMP
conduit.

A working MPI implementation is also required, and your code must be compiled
with `ENABLE_MPIIO_SUPPORT`, if the MPI-IO-based `load` and `save` methods are
desired.

**Note**: In order to use the `load()` and `save()` methods, which use MPI-IO
to read / write distributed matrix data to disk (in hopes of taking advantage
of collective buffering optimizations, etc.), you must first initialize MPI in
your program.

## Feature macros

Enabling production features:

* `-DENABLE_MPIIO_SUPPORT`: enable MPI-IO based `load()` and `save()` methods
  for reading / writing matrix data

## References

[1] Scott French, Yili Zheng, Barbara Romanowicz, Katherine Yelick, "Parallel
Hessian Assembly for Seismic Waveform Inversion Using Global Updates", 29th
IEEE Int. ‘Parallel and Distributed Processing’ Symp., 2015, doi:
[10.1109/IPDPS.2015.58](https://dx.doi.org/10.1109/IPDPS.2015.58).

[2] Scott French, Barbara Romanowicz, "Whole-mantle radially anisotropic shear
velocity structure from spectral-element waveform tomography", Geophysical
Journal International, 2014, 199, doi:
[10.1093/gji/ggu334](https://dx.doi.org/10.1093/gji/ggu334).

[3] Scott French, Barbara Romanowicz, "Broad plumes rooted at the base of the
Earth’s mantle beneath major hotspots", Nature, 2015, 525, doi:
[10.1038/nature14876](https://dx.doi.org/10.1038/nature14876).
