/**
 * \mainpage ConvergentMatrix
 *
 * A "convergent" distributed dense matrix data structure.
 *
 * Model: Additive updates to distributed matrix elements are batched locally
 * and applied asynchronously.
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
 * process. This ensures assumptions surrounding quiescence in methods such as
 * \c commit hold (i.e. operations that ensure user-level progress will execute
 * remotely injected updates), as well as use of collective operations. See the
 * UPC++ Programming Guide or Specification for more details.
 *
 * Thread safety: ConvergentMatrix is not thread safe, however it is thread
 * compatible.
 */

#pragma once

#include <cassert>
#include <cstdlib>

#include <algorithm>
#include <chrono>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <unordered_map>

#include <upcxx/upcxx.hpp>

#ifdef ENABLE_MPIIO_SUPPORT
#include <mpi.h>
#endif

#include "local_matrix.hpp"  // LocalMatrix<T>

// default bin-size threshold (number of elems) before it is flushed
#ifndef DEFAULT_BIN_FLUSH_THRESHOLD
#define DEFAULT_BIN_FLUSH_THRESHOLD 10000
#endif

// default number of update() calls to trigger progress()
#ifndef DEFAULT_PROGRESS_INTERVAL
#define DEFAULT_PROGRESS_INTERVAL 1
#endif

#define CM_LOG                                                         \
  std::cerr << __FILE__ << ":" << __LINE__ << " " << __func__ << " @ " \
            << upcxx::rank_me() << "] "

/**
 * Contains classes associated with the convergent-matrix abstraction
 */
namespace cm {

#ifdef ENABLE_MPIIO_SUPPORT
namespace {
template <typename E>
MPI_Datatype CM_get_mpi_base_type();

template <>
MPI_Datatype CM_get_mpi_base_type<int>() {
  return MPI_INT;
}

template <>
MPI_Datatype CM_get_mpi_base_type<float>() {
  return MPI_FLOAT;
}

template <>
MPI_Datatype CM_get_mpi_base_type<double>() {
  return MPI_DOUBLE;
}
}  // namespace
#endif  // ENABLE_MPIIO_SUPPORT

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
  // matrix dimensions and process grid
  const long _m, _n;              // matrix global dimensions
  const long _myrow, _mycol;      // coordinate in the process grid
  const long _m_local, _n_local;  // local-storage minimum dimensions

  // controls on update() behavior
  int _progress_interval = DEFAULT_PROGRESS_INTERVAL;
  int _bin_flush_threshold = DEFAULT_BIN_FLUSH_THRESHOLD;
  int _flush_counter = 0;

  // single container for distributed data
  struct DistData {
    long updates_applied;           // number of updates applied
    upcxx::global_ptr<T> elements;  // matrix elements
    DistData(long size)
        : updates_applied(0), elements(upcxx::new_array<T>(size)) {}
  };
  upcxx::dist_object<DistData> _dist_data;

  // cache for remote storage pointers fetched from _dist_data
  std::unordered_map<int /*rank*/, upcxx::global_ptr<T>> _elements_cached;

  // single container for batching / dispatching remote updates
  struct Bin {
    int rank;                // target rank
    long updates_sent;       // number of updates sent
    std::vector<T> data;     // update element value
    std::vector<long> inds;  // update element linear index
    Bin(int r) : rank(r), updates_sent(0) {}
    long size() const { return data.size(); }
    void append(T val, long ij) {
      data.push_back(val);
      inds.push_back(ij);
    }
    void flush(upcxx::dist_object<DistData> &ddata) {
      if (size() == 0) return;
      upcxx::rpc_ff(rank, upcxx::source_cx::as_buffered(),
                    [](upcxx::dist_object<DistData> &ddata, long n,
                       upcxx::view<T> data, upcxx::view<long> inds) {
                      T *local = ddata->elements.local();
                      for (long i = 0; i < n; ++i) local[inds[i]] += data[i];
                      ++ddata->updates_applied;
                    },
                    ddata, size(), upcxx::make_view(data),
                    upcxx::make_view(inds));
      ++updates_sent;
      data.clear();
      inds.clear();
    }
  };
  std::vector<std::unique_ptr<Bin>> _bins;
  std::vector<Bin *> _flush_order;  // elements owned by _bins

  using wt_clock_t = std::chrono::high_resolution_clock;
  using wt_t = std::chrono::time_point<wt_clock_t>;
#ifdef ENABLE_UPDATE_TIMING
  wt_t _wt_init;
#endif

  // *********************
  // ** private methods **
  // *********************

  // flush all bins above size thresh.
  void flush(int thresh = 0) {
    for (int i = 0; i < NPROW * NPCOL; ++i) {
      if (_bins[i]->size() > thresh) _bins[i]->flush(_dist_data);
    }

    if (++_flush_counter == _progress_interval || thresh == 0) {
      upcxx::progress();  // user-level progress: may execute injected updates
      _flush_counter = 0;
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
        // setup block-cyclic distribution
        _myrow(upcxx::rank_me() / NPROW),
        _mycol(upcxx::rank_me() % NPCOL),
        // calculate minimum req'd local dimensions
        _m_local(roc(_m, NPROW, _myrow, MB)),
        _n_local(roc(_n, NPCOL, _mycol, NB)),
        // collective: dist_object initialization
        _dist_data(LLD * _n_local) {
    // checks on matrix dimension
    assert(_m > 0);
    assert(_n > 0);

    // check on block-cyclic distribution
    assert(NPCOL * NPROW == upcxx::rank_n());

    // ensure local storage is of nonzero size
    assert(_m_local > 0);
    assert(_n_local > 0);

    // check minimum local leading dimension compatible with LLD
    assert(_m_local <= LLD);

    // initialize distributed matrix data
    T *local_ptr = _dist_data->elements.local();
    std::fill(local_ptr, local_ptr + LLD * _n_local, 0);

    // initialize update bins, providing a randomized flush order to avoid
    // synchronized sweeps across peers
    for (int rank = 0; rank < NPROW * NPCOL; rank++) {
      auto bin = std::make_unique<Bin>(rank);
      _flush_order.push_back(bin.get());
      _bins.push_back(std::move(bin));
    }
    std::random_device rdev;
    std::mt19937 rgen(rdev());
    std::shuffle(_flush_order.begin(), _flush_order.end(), rgen);

#ifdef ENABLE_UPDATE_TIMING
    _wt_init = wt_clock_t::now();
#endif
  }

  /**
   * Destructor for the \c ConvergentMatrix distributed matrix abstraction.
   *
   * \b Note: This destructor is a collective operation, as it performs once
   * final \c commit() call in order to clear remaining injected updates before
   * the underlying distributed matrix storage is freed.
   */
  ~ConvergentMatrix() {
    commit();
    upcxx::delete_array(_dist_data->elements);
  }

  /**
   * Get a raw pointer to the local distributed matrix storage (can be passed
   * to, for example, PBLAS routines).
   *
   * \b Note: The underlying storage _will_ be freed in the \c ConvergentMatrix
   * destructor - for a persistent copy, see \c get_local_data_copy().
   */
  T *get_local_data() const { return _dist_data->elements.local(); }

  /**
   * Get a point to a _copy_ of the local distributed matrix storage (can be
   * passed to, for example, PBLAS routines).
   *
   * \b Note: The underlying storage will _not_ be freed in the \c
   * ConvergentMatrix destructor, in contrast to that from \c get_local_data().
   */
  T *get_local_data_copy() const {
    T *copy_ptr = new T[LLD * _n_local];
    T *local_ptr = _dist_data->elements.local();
    std::copy(local_ptr, local_ptr + LLD * _n_local, copy_ptr);
    return copy_ptr;
  }

  /**
   * Reset the distributed matrix (collective).
   *
   * Calls \c commit() to clear remaining injected updates and then zeros the
   * associated local storage. Returns when all ranks have completed zero-fill.
   */
  void reset() {
    // commit to complete any outstanding updates
    commit();

    // zero local storage
    T *local_ptr = _dist_data->elements.local();
    std::fill(local_ptr, local_ptr + LLD * _n_local, 0);

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
    int rank = (jx / NB) % NPCOL + NPCOL * ((ix / MB) % NPROW);
    long ij = LLD * ((jx / (NB * NPCOL)) * NB + jx % NB) +
              (ix / (MB * NPROW)) * MB + ix % MB;
    auto it = _elements_cached.find(rank);
    if (it == _elements_cached.end()) {
      auto ddata = _dist_data.fetch(rank).wait();
      auto ins = _elements_cached.insert({rank, ddata.elements});
      it = ins.first;
    }
    return upcxx::rget(it->second + ij).wait();
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
        _bins[rank]->append((*Mat)(i, j), ij);
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
    _bins[rank]->append(elem, ij);
    flush(_bin_flush_threshold);
  }

  /**
   * Distributed matrix update: symmetric case (assumes the upper triangular
   * part of the provided \c LocalMatrix<T> has been populated).
   * \param Mat The update (strided) slice
   * \param ix Maps slice into distributed matrix (both dimensions)
   *
   * Populates only the upper triangular part of the distributed matrix. See \c
   * fill_lower() (collective) which may be used to initialize the lower
   * triangular part after \c commit().
   */
  void update(LocalMatrix<T> *Mat, long *ix) {
#ifndef NOCHECK
    // must be square to be symmetric
    assert(Mat->m() == Mat->n());
#endif

#ifdef ENABLE_UPDATE_TIMING
    auto wt0 = wt_clock_t::now();
#endif
    for (long j = 0; j < Mat->n(); j++) {
      int pcol = (ix[j] / NB) % NPCOL;
      long off_j = LLD * ((ix[j] / (NB * NPCOL)) * NB + ix[j] % NB);
      for (long i = 0; i < Mat->m(); i++)
        if (ix[i] <= ix[j]) {
          int rank = pcol + NPCOL * ((ix[i] / MB) % NPROW);
          long ij = off_j + (ix[i] / (MB * NPROW)) * MB + ix[i] % MB;
          _bins[rank]->append((*Mat)(i, j), ij);
        }
    }
#ifdef ENABLE_UPDATE_TIMING
    auto wt1 = wt_clock_t::now();
    {
      std::chrono::duration<double> elapsed = wt1 - _wt_init;
      std::chrono::duration<double> binning = wt1 - wt0;
      CM_LOG << "elapsed time: " << elapsed.count()
             << " s binning time: " << binning.count() << " s" << std::endl;
    }
    wt0 = wt_clock_t::now();
#endif
    flush(_bin_flush_threshold);
#ifdef ENABLE_UPDATE_TIMING
    wt1 = wt_clock_t::now();
    {
      std::chrono::duration<double> elapsed = wt1 - _wt_init;
      std::chrono::duration<double> flush = wt1 - wt0;
      CM_LOG << "elapsed time: " << elapsed.count()
             << " s flush time: " << flush.count() << " s" << std::endl;
    }
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
    T *local_ptr = _dist_data->elements.local();
    for (long j = 0; j < _n_local; j++) {
      long ix = (j / NB) * (NPCOL * NB) + _mycol * NB + j % NB;
      for (long i = 0; i < _m_local; i++) {
        long jx = (i / MB) * (NPROW * MB) + _myrow * MB + i % MB;
        // use _transposed_ global indices to fill in the strict lower part
        if (ix > jx) {
          int rank = (jx / NB) % NPCOL + NPCOL * ((ix / MB) % NPROW);
          long ij = LLD * ((jx / (NB * NPCOL)) * NB + jx % NB) +
                    (ix / (MB * NPROW)) * MB + ix % MB;
          _bins[rank]->append(local_ptr[i + j * LLD], ij);
        } else {
          break;  // nothing left to do in this column ...
        }
      }
    }
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
    // flush any remaining non-empty bins (i.e. size threshold is zero), after
    // which we know that our local updates-sent counts are fully up to date.
    flush();

    // achieve quiescence: spin in progress until the locally applied number of
    // updates matches the expected number of globally injected updates.
    std::vector<long> updates_expected;
    for (const auto &bin : _bins) {
      updates_expected.push_back(bin->updates_sent);
    }
    upcxx::reduce_all(updates_expected.data(), updates_expected.data(),
                      updates_expected.size(), upcxx::op_fast_add)
        .wait();
    const long expected = updates_expected[upcxx::rank_me()];
    while (_dist_data->updates_applied < expected) upcxx::progress();

    // sync: return when *all* ranks have entered quiescence
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
    T *local_ptr = _dist_data->elements.local();
    for (long j = 0; j < _n; j++)
      if ((j / NB) % NPCOL == _mycol) {
        long off_j = LLD * ((j / (NB * NPCOL)) * NB + j % NB);
        for (long i = 0; i < _m; i++)
          if ((i / MB) % NPROW == _myrow) {
            long ij = off_j + (i / (MB * NPROW)) * MB + i % MB;
            if (!eq(local_ptr[ij], i, j)) return false;
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
    MPI_Datatype base_dtype = CM_get_mpi_base_type<T>();
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
    T *local_ptr = _dist_data->elements.local();
    if (_m_local < LLD)
      for (long j = 1; j < _n_local; j++)
        for (long i = 0; i < _m_local; i++)
          local_ptr[i + j * _m_local] = local_ptr[i + j * LLD];

    // write out local data
    auto start = wt_clock_t::now();
    MPI_File_write_all(f_ata, local_ptr, _m_local * _n_local, base_dtype,
                       &status);
    std::chrono::duration<double> wt_io_duration = wt_clock_t::now() - start;

    // close; report io time
    MPI_File_close(&f_ata);
    double wt_io_max, wt_io = wt_io_duration.count();
    MPI_Reduce(&wt_io, &wt_io_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (mpi_rank == 0)
      CM_LOG << "max time spent in matrix write: " << wt_io_max << " s"
             << std::endl;

    // sanity check on data written
    MPI_Get_count(&status, base_dtype, &write_count);
    assert(write_count == (_m_local * _n_local));

    // expansion in place
    if (_m_local < LLD)
      for (long j = _n_local - 1; j > 0; j--)
        for (long i = _m_local - 1; i >= 0; i--)
          local_ptr[i + j * LLD] = local_ptr[i + j * _m_local];

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
    MPI_Datatype base_dtype = CM_get_mpi_base_type<T>();
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
    T *local_ptr = _dist_data->elements.local();
    auto start = wt_clock_t::now();
    MPI_File_read_all(f_ata, local_ptr, _m_local * _n_local, base_dtype,
                      &status);
    std::chrono::duration<double> wt_io_duration = wt_clock_t::now() - start;

    // close; report io time
    MPI_File_close(&f_ata);
    double wt_io_max, wt_io = wt_io_duration.count();
    MPI_Reduce(&wt_io, &wt_io_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (mpi_rank == 0)
      CM_LOG << "max time spent in matrix read: " << wt_io_max << " s"
             << std::endl;

    // sanity check on data read
    MPI_Get_count(&status, base_dtype, &read_count);
    assert(read_count == (_m_local * _n_local));

    // expansion in place
    if (_m_local < LLD)
      for (long j = _n_local - 1; j > 0; j--)
        for (long i = _m_local - 1; i >= 0; i--)
          local_ptr[i + j * LLD] = local_ptr[i + j * _m_local];

    // sync post expansion (ops on distributed matrix elems can start again)
    upcxx::barrier();

    // free distributed type
    MPI_Type_free(&distmat);
  }

#endif  // ENABLE_MPIIO_SUPPORT

};  // end of ConvergentMatrix

}  // end of namespace cm
