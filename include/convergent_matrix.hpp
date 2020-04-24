/**
 * \mainpage ConvergentMatrix
 *
 * A "convergent" distributed dense matrix data structure.
 *
 * Key components:
 *  - The \c ConvergentMatrix<T,NPROW,NPCOL,MB,NB,LLD> abstraction accumulates
 *    updates to the global distributed matrix in bins (\c Bin<T>) for later
 *    asynchronous application.
 *  - The \c Bin<T> object implements the binning concept in \c
 *    ConvergentMatrix, and handles "flushing" its contents by triggering
 *    remote updates (\c update_task<T>) via \c upcxx::rpc calls.
 *
 * Matrix updates may target either a single distributed matrix element or a
 * "slice" of the distributed matrix, with the latter represented by a \c
 * LocalMatrix<T> instance (see local_matrix.hpp) along with indexing arrays
 * which map into the global distributed index space.
 *
 * Once a series of updates have been initiated (see \c update), the matrix can
 * be "committed" (see \c commit). Once \c commit returns, it is guaranteed
 * that all previously requested remote updated have been applied. Multiple
 * successive "rounds" of \c update and \c commit calls are permitted.
 *
 * After \c commit returns, each process has its own PBLAS-compatible portion
 * of the global matrix, consistent with the block-cyclic distribution defined
 * by the template parameters of \c ConvergentMatrix and assuming a row-major
 * order of processes in the process grid.
 *
 * Progress: In general, it is assumed that \c ConvergentMatrix instances are
 * only manipulated by the thread holding the master persona on a participating
 * process. This ensures that assumptions surrounding quiescence in methods
 * such as \c commit hold (i.e. entering operations that ensure user-level
 * progress therein will execute remotely injected updates). See the UPC++
 * Programming Guide or Specification for more details.
 *
 * Thread safety: ConvergentMatrix is not thread safe, however it is thread
 * compatible.
 *
 * \b Note: by default, no documentation is produced for internal data
 * structures and functions (e.g. \c update_task<T> and \c Bin<T>). To enable
 * internal documentation, add \c INTERNAL_DOCS to the \c ENABLED_SECTIONS
 * option in \c doxygen.conf.
 */

#pragma once

#include <cassert>
#include <cstdio>
#include <cstdlib>

#include <algorithm>
#include <functional>
#include <memory>
#include <random>

#include <upcxx/upcxx.hpp>

#ifdef ENABLE_MPIIO_SUPPORT
#include <mpi.h>
#define ENABLE_MPI_HELPERS
#endif

#ifdef ENABLE_UPDATE_TIMING
#include <omp.h>
#endif

// cm additions / internals
#include "bin.hpp"           // Bin<T>
#include "local_matrix.hpp"  // LocalMatrix<T>

// default bin-size threshold (number of elems) before it is flushed
#ifndef DEFAULT_BIN_FLUSH_THRESHOLD
#define DEFAULT_BIN_FLUSH_THRESHOLD 10000
#endif

// default number of update() calls to trigger progress()
#ifndef DEFAULT_PROGRESS_INTERVAL
#define DEFAULT_PROGRESS_INTERVAL 1
#endif

/**
 * Contains classes associated with the convergent-matrix abstraction
 */
namespace cm {

/**
 * Convergent matrix abstraction
 * \tparam T Matrix data type (e.g. float)
 * \tparam NPROW Number of rows in the distributed process grid
 * \tparam NPCOL Number of columns in the distributed process grid
 * \tparam MB Distribution blocking factor (leading dimension)
 * \tparam NB Distribution blocking factor (trailing dimension)
 * \tparam LLD Leading dimension of local storage (same on all processes)
 */
template <typename T,              // matrix type
          long NPROW, long NPCOL,  // pblas process grid
          long MB, long NB,        // pblas local blocking factors
          long LLD>                // pblas local leading dim
class ConvergentMatrix {
 private:
  // matrix dimension and process grid
  long _m, _n;              // matrix global dimensions
  long _m_local, _n_local;  // local-storage minimum dimensions
  long _myrow, _mycol;      // coordinate in the process grid

  // update binning and application
  int _progress_interval;
  int _bin_flush_threshold;
  int _flush_counter;
  int _bin_flush_order[NPROW * NPCOL];
  Bin<T> _update_bins[NPROW * NPCOL];

  // promise for this epoch.
  std::unique_ptr<upcxx::promise<>> _curr_promise;

  // distributed storage arrays (local and remote)
  T *_local_ptr;
  upcxx::global_ptr<T> _g_local_ptr;
  upcxx::global_ptr<T> _remote_ptrs[NPROW * NPCOL];

#ifdef ENABLE_UPDATE_TIMING
  double _wt_init;
#endif

  // *********************
  // ** private methods **
  // *********************

  void init_arrays() {
    // allocate GASNet addressable storage and initialize.
    _g_local_ptr = upcxx::new_array<T>(LLD * _n_local);
    _local_ptr = _g_local_ptr.local();
    std::fill(_local_ptr, _local_ptr + LLD * _n_local, 0);

    // temporary for exchanging pointers (note: ctor is a collective).
    upcxx::dist_object<upcxx::global_ptr<T>> g_tmp(_g_local_ptr);

    // sync: ensure dist_object ctor complete on all processes.
    upcxx::barrier();

    // copy remote pointers into _remote_ptrs, ensuring that future use of
    // remote pointers does not incur a dist_object::fetch call.
    upcxx::future<> fut_all = upcxx::make_future();
    for (int rank = 0; rank < NPROW * NPCOL; ++rank) {
      upcxx::future<> fut = g_tmp.fetch(rank).then(
          [rank, this](upcxx::global_ptr<T> ptr) { _remote_ptrs[rank] = ptr; });
      fut_all = upcxx::when_all(fut_all, fut);
    }
    fut_all.wait();

    // sync: ensure all processes have obtained remote pointers before g_tmp
    // goes out of scope.
    upcxx::barrier();
  }

  void init_bins() {
    for (int rank = 0; rank < NPROW * NPCOL; rank++)
      _update_bins[rank].init(_remote_ptrs[rank]);

    // randomize flush order to avoid synchronized sweeps across all
    // participants.
    for (int rank = 0; rank < NPROW * NPCOL; rank++)
      _bin_flush_order[rank] = rank;
    std::random_device rdev;
    std::mt19937 rgen(rdev());
    std::shuffle(_bin_flush_order, _bin_flush_order + (NPROW * NPCOL), rgen);
  }

  void flush(int thresh = 0) {
    for (int bin = 0; bin < NPROW * NPCOL; bin++) {
      int rank = _bin_flush_order[bin];
      if (_update_bins[rank].size() > thresh)
        _update_bins[rank].flush(_curr_promise.get());
    }

    if (thresh == 0 || ++_flush_counter == _progress_interval) {
      upcxx::progress();  // user-level progress: may execute injected updates
      _flush_counter = 0;
    } else {
      // at least invoke internal-level progress
      upcxx::progress(upcxx::progress_level::internal);
    }
  }

  // determine lower bound on local storage for a given block-cyclic
  // distribution
  long roc(long m,   // matrix dimension
           long np,  // dimension of the process grid
           long ip,  // index in the process grid
           long mb)  // blocking factor
  {
    long nblocks, m_local;
    // number of full blocks to be distributed
    nblocks = m / mb;
    // set lower bound on required local dimension
    m_local = (nblocks / np) * mb;
    // edge cases ...
    if (nblocks % np > ip)
      m_local += mb;  // process ip receives another full block
    else if (nblocks % np == ip)
      m_local += m % mb;  // process ip _may_ receive a partial block
    return m_local;
  }

#ifdef ENABLE_MPI_HELPERS

  MPI_Datatype get_mpi_base_type(int *) { return MPI_INT; }

  MPI_Datatype get_mpi_base_type(float *) { return MPI_FLOAT; }

  MPI_Datatype get_mpi_base_type(double *) { return MPI_DOUBLE; }

  MPI_Datatype get_mpi_base_type() { return get_mpi_base_type(_local_ptr); }

#endif  // ENABLE_MPI_HELPERS

 public:
  // ********************
  // ** public methods **
  // ********************

  /**
   * Initialize the \c ConvergentMatrix distributed matrix abstraction.
   * \param m Global leading dimension of the distributed matrix
   * \param n Global trailing dimension of the distributed matrix
   *
   * \b Note: This constructor is a collective operation.
   */
  ConvergentMatrix(long m, long n)
      : _m(m),
        _n(n),
        _progress_interval(DEFAULT_PROGRESS_INTERVAL),
        _bin_flush_threshold(DEFAULT_BIN_FLUSH_THRESHOLD),
        _flush_counter(0) {
    // checks on matrix dimension
    assert(_m > 0);
    assert(_n > 0);

    // check on block-cyclic distribution
    assert(NPCOL * NPROW == upcxx::rank_n());

    // setup block-cyclic distribution
    _myrow = upcxx::rank_me() / NPROW;
    _mycol = upcxx::rank_me() % NPCOL;

    // calculate minimum req'd local dimensions
    _m_local = roc(_m, NPROW, _myrow, MB);
    _n_local = roc(_n, NPCOL, _mycol, NB);

    // ensure local storage is of nonzero size
    assert(_m_local > 0);
    assert(_n_local > 0);

    // check minimum local leading dimension compatible with LLD
    assert(_m_local <= LLD);

    // initialize distributed storage
    init_arrays();

    // initialize update bins
    init_bins();

    // initialize the promise that will track remote updates during this epoch
    // (i.e. until the next call to commit()).
    _curr_promise = std::make_unique<upcxx::promise<>>();

#ifdef ENABLE_UPDATE_TIMING
    _wt_init = omp_get_wtime();
#endif
  }

  /**
   * Destructor for the \c ConvergentMatrix distributed matrix abstraction.
   *
   * On destruction, will:
   * - Perform a final \c commit(), clearing remaining injected updates from
   *   the queue
   * - Free storage associated with update bins
   * - Free storage associated with the distributed matrix
   */
  ~ConvergentMatrix() {
    commit();

    // delete the GASNet-addressable local storage
    upcxx::delete_array(_g_local_ptr);
  }

  /**
   * Get a raw pointer to the local distributed matrix storage (can be passed
   * to, for example, PBLAS routines).
   *
   * \b Note: The underlying storage _will_ be freed in the \c ConvergentMatrix
   * destructor - for a persistent copy, see \c get_local_data_copy().
   */
  T *get_local_data() const { return _local_ptr; }

  /**
   * Get a point to a _copy_ of the local distributed matrix storage (can be
   * passed to, for example, PBLAS routines).
   *
   * \b Note: The underlying storage will _not_ be freed in the
   * \c ConvergentMatrix destructor, in contrast to that from
   * \c get_local_data().
   */
  T *get_local_data_copy() const {
    T *copy_ptr = new T[LLD * _n_local];
    std::copy(_local_ptr, _local_ptr + LLD * _n_local, copy_ptr);
    return copy_ptr;
  }

  /**
   * Reset the distributed matrix (implicit barrier)
   *
   * Calls \c commit(), clearing remaining injected updates from the queue, and
   * then zeros the associated local storage.
   */
  void reset() {
    // commit to complete any outstanding updates
    commit();

    // zero local storage
    std::fill(_local_ptr, _local_ptr + LLD * _n_local, 0);

    // synchronize
    upcxx::barrier();
  }

  /**
   * Get the flush threshold (maximum bulk-update bin size before a bin is
   * flushed and applied to its target).
   */
  int bin_flush_threshold() const { return _bin_flush_threshold; }

  /**
   * Set the flush threshold (maximum bulk-update bin size before a bin is
   * flushed and applied to its target).
   * \param thresh The bin-size threshold
   */
  void bin_flush_threshold(int thresh) { _bin_flush_threshold = thresh; }

  /**
   * Get the progress interval, the number of bulk-update bin-flushes before
   * calling into upcxx::progress to ensure completion of remotely injected
   * updates.
   */
  int progress_interval() const { return _progress_interval; }

  /**
   * Set the progress interval, the number of bulk-update bin-flushes before
   * calling into upcxx::progress to ensure completion of remotely injected
   * updates.
   * \param interval The progress interval
   */
  void progress_interval(int interval) { _progress_interval = interval; }

  /**
   * Distributed matrix leading dimension
   */
  long m() const { return _m; }

  /**
   * Distributed matrix trailing dimension
   */
  long n() const { return _n; }

  /**
   * Process grid row index of this process
   */
  long pgrid_row() const { return _myrow; }

  /**
   * Process grid column index of this process
   */
  long pgrid_col() const { return _mycol; }

  /**
   * Minimum required leading dimension of local storage - must be less than
   * or equal to template parameter LLD
   */
  long m_local() const { return _m_local; }

  /**
   * Minimum required trailing dimension of local storage
   */
  long n_local() const { return _n_local; }

  /**
   * Remote random access (read only) to distributed matrix elements
   * \param ix Leading dimension index
   * \param jx Trailing dimension index
   */
  T operator()(long ix, long jx) {
    // infer process rank and linear index
    int rank = (jx / NB) % NPCOL + NPCOL * ((ix / MB) % NPROW);
    long ij = LLD * ((jx / (NB * NPCOL)) * NB + jx % NB) +
              (ix / (MB * NPROW)) * MB + ix % MB;
    return upcxx::rget(_remote_ptrs[rank] + ij).wait();
  }

  /**
   * Distributed matrix update: general case
   * \param Mat The update (strided) slice
   * \param ix Maps slice into distributed matrix (leading dimension)
   * \param jx Maps slice into distributed matrix (trailing dimension)
   */
  void update(LocalMatrix<T> *Mat, long *ix, long *jx) {
    for (long j = 0; j < Mat->n(); j++) {
      int pcol = (jx[j] / NB) % NPCOL;
      long off_j = LLD * ((jx[j] / (NB * NPCOL)) * NB + jx[j] % NB);
      for (long i = 0; i < Mat->m(); i++) {
        int rank = pcol + NPCOL * ((ix[i] / MB) % NPROW);
        long ij = off_j + (ix[i] / (MB * NPROW)) * MB + ix[i] % MB;
        _update_bins[rank].append((*Mat)(i, j), ij);
      }
    }

    flush(_bin_flush_threshold);
  }

  /**
   * Distributed matrix update: general _elemental_ case
   * \param elem The elemental update
   * \param ix Global index in distributed matrix (leading dimension)
   * \param jx Global index in distributed matrix (trailing dimension)
   */
  void update(T elem, long ix, long jx) {
    int rank = (jx / NB) % NPCOL + NPCOL * ((ix / MB) % NPROW);
    long ij = LLD * ((jx / (NB * NPCOL)) * NB + jx % NB) +
              (ix / (MB * NPROW)) * MB + ix % MB;
    _update_bins[rank].append(elem, ij);

    flush(_bin_flush_threshold);
  }

  /**
   * Distributed matrix update: symmetric case
   * \param Mat The update (strided) slice
   * \param ix Maps slice into distributed matrix (both dimensions)
   */
  void update(LocalMatrix<T> *Mat, long *ix) {
#ifdef ENABLE_UPDATE_TIMING
    double wt0, wt1;
#endif
#ifndef NOCHECK
    // must be square to be symmetric
    assert(Mat->m() == Mat->n());
#endif

#ifdef ENABLE_UPDATE_TIMING
    wt0 = omp_get_wtime();
#endif
    // bin the local update
    for (long j = 0; j < Mat->n(); j++) {
      int pcol = (ix[j] / NB) % NPCOL;
      long off_j = LLD * ((ix[j] / (NB * NPCOL)) * NB + ix[j] % NB);
      for (long i = 0; i < Mat->m(); i++)
        if (ix[i] <= ix[j]) {
          int rank = pcol + NPCOL * ((ix[i] / MB) % NPROW);
          long ij = off_j + (ix[i] / (MB * NPROW)) * MB + ix[i] % MB;
          _update_bins[rank].append((*Mat)(i, j), ij);
        }
    }
#ifdef ENABLE_UPDATE_TIMING
    wt1 = omp_get_wtime();
    printf("[%s] %f binning time: %f s\n", __func__, wt1 - _wt_init, wt1 - wt0);
#endif

#ifdef ENABLE_UPDATE_TIMING
    wt0 = omp_get_wtime();
#endif
    flush(_bin_flush_threshold);
#ifdef ENABLE_UPDATE_TIMING
    wt1 = omp_get_wtime();
    printf("[%s] %f flush time: %f s\n", __func__, wt1 - _wt_init, wt1 - wt0);
#endif
  }

  /**
   * Fill in the lower triangular part of a symmetric distributed matrix in
   * a single sweep.
   *
   * This routine uses the same logic as the general update case and only
   * ensures that the requisite updates have been initiated on the source
   * side.
   *
   * \b Note: There is also no implicit \c commit() before the fill updates
   * start. Thus, always call \c commit() _before_ calling \c fill_lower(),
   * and again when you need to ensure the full updates have been applied.
   */
  void fill_lower() {
    for (long j = 0; j < _n_local; j++) {
      long ix = (j / NB) * (NPCOL * NB) + _mycol * NB + j % NB;
      for (long i = 0; i < _m_local; i++) {
        long jx = (i / MB) * (NPROW * MB) + _myrow * MB + i % MB;
        // use _transposed_ global indices to fill in the strict lower part
        if (ix > jx) {
          int rank = (jx / NB) % NPCOL + NPCOL * ((ix / MB) % NPROW);
          long ij = LLD * ((jx / (NB * NPCOL)) * NB + jx % NB) +
                    (ix / (MB * NPROW)) * MB + ix % MB;
          _update_bins[rank].append(_local_ptr[i + j * LLD], ij);
        } else {
          break;  // nothing left to do in this column ...
        }
      }
    }

    // possibly flush bins
    flush(_bin_flush_threshold);
  }

  /**
   * Drain all update bins and wait on associated update RPCs.
   *
   * Once commit returns, all remote updates previously requested via calls to
   * \c update are guaranteed to have been applied.
   *
   * \b Note: \c commit is a collective operation.
   */
  void commit() {
    // synchronize
    upcxx::barrier();

    // flush all non-empty bins (i.e. bin size threshold is zero).
    flush();

    // sync: ensure all remote update RPCs have been dispatched.
    upcxx::barrier();

    // progress any newly injected RPCs.
    // TODO: We may need a stronger guarantee of quiescence here.
    upcxx::progress();

    // wait on dispatched update RPC completion and reset the promise for our
    // next pass.
    _curr_promise->finalize().wait();
    _curr_promise = std::make_unique<upcxx::promise<>>();

    // wait for all processes to observe completion of dispatched updates (note:
    // barrier() internally makes user-level progress, and will thus execute
    // remotely injected RPCs).
    upcxx::barrier();
  }

  /**
   * Map over locally stored matrix elements, calling \c eq the stored value,
   * row, and column for each. \c eq is expected to compare the element value
   * with an externally maintained source of truth, returning \c false on
   * mismatch.
   *
   * Returns \c false on the first element that fails to verify
   *
   * \param eq Single-element verification function.
   *
   * \b Note: Used only in tests.
   */
  bool verify_local_elements(std::function<bool(const T, long, long)> eq) {
    for (long j = 0; j < _n; j++)
      if ((j / NB) % NPCOL == _mycol) {
        long off_j = LLD * ((j / (NB * NPCOL)) * NB + j % NB);
        for (long i = 0; i < _m; i++)
          if ((i / MB) % NPROW == _myrow) {
            long ij = off_j + (i / (MB * NPROW)) * MB + i % MB;
            if (!eq(_local_ptr[ij], i, j)) return false;
          }
      }
    return true;
  }

  // Optional functionality enabled at compile time ...

#ifdef ENABLE_MPIIO_SUPPORT

  /**
   * Save the distributed matrix to disk via MPI-IO.
   * \param fname File name for matrix
   *
   * \b Note: No implicit \c commit() before matrix data is written - always
   * call \c commit() first.
   *
   * \b Note: Requires compilation with \c ENABLE_MPIIO_SUPPORT and MPI must
   * already have been initialized by the user.
   *
   * \b Note: If compiled with \c THREAD_FUNNELED_REQUIRED, this routine will
   * require MPI to have been initialized with a thread support level of at
   * least \c MPI_THREAD_FUNNELED.
   */
  void save(const char *fname) {
    int mpi_init, mpi_rank, distmat_size, write_count;
#ifdef THREAD_FUNNELED_REQUIRED
    int mpi_thread;
#endif  // THREAD_FUNNELED_REQUIRED
    double wt_io, wt_io_max;
    MPI_Status status;
    MPI_Datatype distmat;
    MPI_File f_ata;

    // make sure we all get here
    upcxx::barrier();

    // make sure MPI is already initialized
    assert(MPI_Initialized(&mpi_init) == MPI_SUCCESS);
    assert(mpi_init);
#ifdef THREAD_FUNNELED_REQUIRED
    // make sure MPI has sufficient thread support
    assert(MPI_Query_thread(&mpi_thread) == MPI_SUCCESS);
    assert(mpi_thread >= MPI_THREAD_FUNNELED);
#endif  // THREAD_FUNNELED_REQUIRED

    // check process grid ordering
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    assert(_myrow * NPCOL + _mycol == mpi_rank);

    // initialize distributed type
    int gsizes[] = {(int)_m, (int)_n},
        distribs[] = {MPI_DISTRIBUTE_CYCLIC, MPI_DISTRIBUTE_CYCLIC},
        dargs[] = {MB, NB}, psizes[] = {NPROW, NPCOL};
    MPI_Datatype base_dtype = get_mpi_base_type();
    MPI_Type_create_darray(NPCOL * NPROW, mpi_rank, 2, gsizes, distribs, dargs,
                           psizes, MPI_ORDER_FORTRAN, base_dtype, &distmat);
    MPI_Type_commit(&distmat);

    // sanity check on check on distributed array data size
    MPI_Type_size(distmat, &distmat_size);
    assert(distmat_size / static_cast<int>(sizeof(T)) == (_m_local * _n_local));

    // open for writing
    MPI_File_open(MPI_COMM_WORLD, fname, MPI_MODE_CREATE | MPI_MODE_WRONLY,
                  MPI_INFO_NULL, &f_ata);

    // set view w/ distmat
    MPI_File_set_view(f_ata, 0, base_dtype, distmat, "native", MPI_INFO_NULL);

    // compaction in place
    if (_m_local < LLD)
      for (long j = 1; j < _n_local; j++)
        for (long i = 0; i < _m_local; i++)
          _local_ptr[i + j * _m_local] = _local_ptr[i + j * LLD];

    // write out local data
    wt_io = -MPI_Wtime();
    MPI_File_write_all(f_ata, _local_ptr, _m_local * _n_local, base_dtype,
                       &status);
    wt_io = wt_io + MPI_Wtime();

    // close; report io time
    MPI_File_close(&f_ata);
    MPI_Reduce(&wt_io, &wt_io_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (mpi_rank == 0)
      printf("[%s] max time spent in matrix write: %.3f s\n", __func__,
             wt_io_max);

    // sanity check on data written
    MPI_Get_count(&status, base_dtype, &write_count);
    assert(write_count == (_m_local * _n_local));

    // expansion in place
    if (_m_local < LLD)
      for (long j = _n_local - 1; j > 0; j--)
        for (long i = _m_local - 1; i >= 0; i--)
          _local_ptr[i + j * LLD] = _local_ptr[i + j * _m_local];

    // sync post expansion (ops on distributed matrix elems can start again)
    upcxx::barrier();

    // free distributed type
    MPI_Type_free(&distmat);
  }

  /**
   * Load a distributed matrix to disk via MPI-IO.
   * \param fname File name for matrix
   *
   * \b Note: Replaces the current contents of the distributed storage array.
   *
   * \b Note: Requires compilation with \c ENABLE_MPIIO_SUPPORT and MPI must
   * already have been initialized by the user.
   *
   * \b Note: If compiled with \c THREAD_FUNNELED_REQUIRED, this routine will
   * require MPI to have been initialized with a thread support level of at
   * least \c MPI_THREAD_FUNNELED.
   */
  void load(const char *fname) {
    int mpi_init, mpi_rank, distmat_size, read_count;
#ifdef THREAD_FUNNELED_REQUIRED
    int mpi_thread;
#endif  // THREAD_FUNNELED_REQUIRED
    double wt_io, wt_io_max;
    MPI_Status status;
    MPI_Datatype distmat;
    MPI_File f_ata;

    // make sure we all get here
    upcxx::barrier();

    // make sure MPI is already initialized
    assert(MPI_Initialized(&mpi_init) == MPI_SUCCESS);
    assert(mpi_init);
#ifdef THREAD_FUNNELED_REQUIRED
    // make sure MPI has sufficient thread support
    assert(MPI_Query_thread(&mpi_thread) == MPI_SUCCESS);
    assert(mpi_thread >= MPI_THREAD_FUNNELED);
#endif  // THREAD_FUNNELED_REQUIRED

    // check process grid ordering
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    assert(_myrow * NPCOL + _mycol == mpi_rank);

    // initialize distributed type
    int gsizes[] = {(int)_m, (int)_n},
        distribs[] = {MPI_DISTRIBUTE_CYCLIC, MPI_DISTRIBUTE_CYCLIC},
        dargs[] = {MB, NB}, psizes[] = {NPROW, NPCOL};
    MPI_Datatype base_dtype = get_mpi_base_type();
    MPI_Type_create_darray(NPCOL * NPROW, mpi_rank, 2, gsizes, distribs, dargs,
                           psizes, MPI_ORDER_FORTRAN, base_dtype, &distmat);
    MPI_Type_commit(&distmat);

    // sanity check on check on distributed array data size
    MPI_Type_size(distmat, &distmat_size);
    assert(distmat_size / static_cast<int>(sizeof(T)) == (_m_local * _n_local));

    // open read-only
    MPI_File_open(MPI_COMM_WORLD, fname, MPI_MODE_RDONLY, MPI_INFO_NULL,
                  &f_ata);

    // set view w/ distmat
    MPI_File_set_view(f_ata, 0, base_dtype, distmat, "native", MPI_INFO_NULL);

    // read in local data
    wt_io = -MPI_Wtime();
    MPI_File_read_all(f_ata, _local_ptr, _m_local * _n_local, base_dtype,
                      &status);
    wt_io = wt_io + MPI_Wtime();

    // close; report io time
    MPI_File_close(&f_ata);
    MPI_Reduce(&wt_io, &wt_io_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (mpi_rank == 0)
      printf("[%s] max time spent in matrix read: %.3f s\n", __func__,
             wt_io_max);

    // sanity check on data read
    MPI_Get_count(&status, base_dtype, &read_count);
    assert(read_count == (_m_local * _n_local));

    // expansion in place
    if (_m_local < LLD)
      for (long j = _n_local - 1; j > 0; j--)
        for (long i = _m_local - 1; i >= 0; i--)
          _local_ptr[i + j * LLD] = _local_ptr[i + j * _m_local];

    // sync post expansion (ops on distributed matrix elems can start again)
    upcxx::barrier();

    // free distributed type
    MPI_Type_free(&distmat);
  }

#endif  // ENABLE_MPIIO_SUPPORT

};  // end of ConvergentMatrix

}  // end of namespace cm
