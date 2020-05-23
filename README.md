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

Since then, `ConvergentMatrix` has been ported to  UPC++ v1.0 and modified to
adopt newly available idioms (e.g. composition of asynchronous operations via
future-based callback cascade).

**Note**: The final release of `ConvergentMatrix` based on the v0.1 spec is
`r20150916`.

## Properties

`ConvergentMatrix` accepts additive updates to distributed matrix elements,
which are buffered locally and applied asynchronously, potentially out of
order (see `update`).

The distributed matrix "converges" to its final state after all matrix updates
have been requested and `commit` is called by all participating processes (a
collective operation). After `commit` returns, the PBLAS-compatible (e.g. for
use with ScaLAPACK) portion of the distributed matrix local to each process is
fully up to date.

### Progress

It is generally assumed that `ConvergentMatrix` instances are manipulated by
the thread holding the master persona on a participating process.

By default, `update` will periodically attempt to make user-level progress in
order to execute remotely injected update RPCs (which is effective only if the
`update` caller holds the master persona), although this may be disabled if the
caller would prefer to manage progress directly (or, e.g., progress has been
delegated to a separate thread holding the master persona).

More importantly, collective operations such as `commit` *must* be called by
the holder of the master persona.

See the UPC++ Programming Guide or Specification for more details on the
progress and persona model, including constraints surrounding RPC execution and
collective operations.

### Thread safety

`ConvergentMatrix` is not thread safe, although it may be considered thread
compatible in certain contexts (and with some care).

Notably, remotely injected update RPCs are not *explicitly* serialized (e.g.
via locks) but are instead *implicitly* serialized through progress guarantees
surrounding dispatch of injected RPCs (i.e. in a valid UPC++ program, only a
single thread associated with a given process will hold the master persona at a
give time, and only user-level progress under the master persona will dispatch
injected RPCs).

In addition, while it is valid for a given program to instantiate multiple
`ConvergentMatrix` instances, care must be taken when reasoning about isolation
between them. For example, methods such as `commit` that invoke user-level
progress may execute remotely injected updates targeting *any* instance (not
just the callee). Thus, it cannot be assumed that access to a given instance
that is concurrent with such calls on *another* instance is safe.

## Dependencies

`ConvergentMatrix` requires a C++ compiler supporting the C++14 standard or
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
