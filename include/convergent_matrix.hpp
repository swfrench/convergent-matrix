/**
 * \mainpage ConvergentMatrix
 *
 * A "convergent" distributed dense matrix data structure
 *
 * Key components:
 *  - The \c ConvergentMatrix<T,NPROW,NPCOL,MB,NB,LLD> abstraction accumulates
 *    updates to the global distributed matrix in bins (\c Bin<T>) for later
 *    asynchronous application.
 *  - The \c Bin<T> object implements the binning concept in \c ConvergentMatrix,
 *    and handles "flushing" its contents by triggering remote asynchronous
 *    updates (\c update_task<T>).
 *
 * Updates are represented as \c LocalMatrix<T> objects (see local_matrix.hpp),
 * along with indexing arrays which map into the global distributed index
 * space.
 *
 * Once a series of updates have been applied, the matrix can be "committed"
 * (see \c ConvergentMatrix::commit()) and all bins are flushed.
 * Thereafter, each thread has its own PBLAS-compatible portion of the global
 * matrix, consistent with the block-cyclic distribution defined by the
 * template parameters of \c ConvergentMatrix and assuming a row-major order of
 * threads in the process grid.
 *
 * \b Note: by default, no documentation is produced for internal data
 * structures and functions (e.g. \c update_task<T> and \c Bin<T>). To enable
 * internal documentation, add \c INTERNAL_DOCS to the \c ENABLED_SECTIONS option
 * in \c doxygen.conf.
 */

#pragma once

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cassert>

#include <algorithm>

#include <upcxx.h>

#ifdef ENABLE_CONSISTENCY_CHECK
#include <cmath>
#endif

#ifdef ENABLE_PROGRESS_THREAD
#include <pthread.h>
#include <unistd.h>  // usleep
#define PROGRESS_HELPER_PAUSE_USEC 1000
#endif

#if ( defined(ENABLE_CONSISTENCY_CHECK) || \
      defined(ENABLE_MPIIO_SUPPORT) )
#include <mpi.h>
#define ENABLE_MPI_HELPERS
#endif

#ifdef ENABLE_UPDATE_TIMING
#include <omp.h>
#endif

// cm additions / internals
#include "local_matrix.hpp"  // LocalMatrix<T>
#include "bin.hpp"           // Bin<T>

// default bin-size threshold (number of elems) before it is flushed
#ifndef DEFAULT_BIN_FLUSH_THRESHOLD
#define DEFAULT_BIN_FLUSH_THRESHOLD 10000
#endif

// default number of update() to trigger progress() and drain (local) task queue
#ifndef DEFAULT_PROGRESS_INTERVAL
#define DEFAULT_PROGRESS_INTERVAL 1
#endif


/**
 * Contains classes associated with the convergent-matrix abstraction
 */
namespace cm
{

  /// @cond INTERNAL_DOCS

#ifdef ENABLE_PROGRESS_THREAD

  /**
   * Argument struct for the \c progress_helper() thread
   */
  struct progress_helper_args
  {
    // lock for operations on the task queue
    pthread_mutex_t * tq_mutex;

    // boolean to signal the progres thread to exit
    bool *progress_thread_stop;
  };

  /**
   * The action performed by the \c progress_helper() thread
   * \param args_ptr A \c void type pointer to a progress_helper_args structure
   */
  void *
  progress_helper( void *args_ptr )
  {
    // re-cast args ptr
    progress_helper_args *args = (progress_helper_args *)args_ptr;

    // spin in upcxx::drain()
    while ( 1 ) {
      pthread_mutex_lock( args->tq_mutex );

      // check the stop flag, possibly exiting
      if ( *args->progress_thread_stop ) {
        pthread_mutex_unlock( args->tq_mutex );
        delete args;
        return NULL;
      }

      // drain the task queue
      upcxx::drain();

      pthread_mutex_unlock( args->tq_mutex );

      // pause briefly
      usleep( PROGRESS_HELPER_PAUSE_USEC );
    }
  }

#endif // ENABLE_PROGRESS_THREAD

  /**
   * Wrapper for \c std::rand() for use in \c std::random_shuffle() (called in
   * \c permute() below).
   */
  int
  rgen( int n )
  {
    return std::rand() % n;
  }

  /**
   * Apply a random permutation (in place) to the supplied array.
   * \param xs Array of type \c int
   * \param nx Length of array \c xs
   *
   * Seeds std::rand() in a manner that should be ok even if all threads hit
   * \c permute() at nearly the same time.
   */
  void
  permute( int *xs, int n )
  {
    unsigned seed = ( 1 + MYTHREAD ) * std::time( NULL );
    std::srand( seed );
    std::random_shuffle( xs, xs + n, rgen );
  }

  /// @endcond

  /**
   * Convergent matrix abstraction
   * \tparam T Matrix data type (e.g. float)
   * \tparam NPROW Number of rows in the distributed process grid
   * \tparam NPCOL Number of columns in the distributed process grid
   * \tparam MB Distribution blocking factor (leading dimension)
   * \tparam NB Distribution blocking factor (trailing dimension)
   * \tparam LLD Leading dimension of local storage (same on all threads)
   */
  template <typename T,              // matrix type
            long NPROW, long NPCOL,  // pblas process grid
            long MB, long NB,        // pblas local blocking factors
            long LLD>                // pblas local leading dim
  class ConvergentMatrix
  {

   private:

    // matrix dimension and process grid
    long _m, _n;              // matrix global dimensions
    long _m_local, _n_local;  // local-storage minimum dimensions
    long _myrow, _mycol;      // coordinate in the process grid

    // update binning and application
    int _flush_counter;
    int _progress_interval;
    int _bin_flush_threshold;
    int _bin_flush_order[NPROW * NPCOL];
    Bin<T> _update_bins[NPROW * NPCOL];
    upcxx::event _e_update;

    // distributed storage arrays (local and remote)
    T *_local_ptr;
    upcxx::global_ptr<T> _remote_ptrs[NPROW * NPCOL];

#ifdef ENABLE_UPDATE_TIMING
    double _wt_init;
#endif

    // replicated consistency checks
#ifdef ENABLE_CONSISTENCY_CHECK
    bool _consistency_mode;
    LocalMatrix<T> * _update_record;
#endif

    // background progress threads
#ifdef ENABLE_PROGRESS_THREAD
    bool _progress_thread_stop, _progress_thread_running;
    pthread_t _progress_thread;
    pthread_mutex_t _tq_mutex;
#endif

    // *********************
    // ** private methods **
    // *********************

    // initialize distributed storage arrays
    inline void
    init_arrays()
    {
      // initialize shared_array of global_ptrs
      upcxx::shared_array<upcxx::global_ptr<T> > shared_remote_ptrs;
      shared_remote_ptrs.init( NPROW * NPCOL );

      // allocate GASNet addressable storage
      upcxx::global_ptr<T> _g_local_ptr =
        upcxx::allocate<T>( MYTHREAD, LLD * _n_local );

      // cast to local ptr and zero
      _local_ptr = (T *) _g_local_ptr;
      std::fill( _local_ptr, _local_ptr + LLD * _n_local, 0 );

      // exchange our global_ptr ref with the other upcxx threads
      shared_remote_ptrs[MYTHREAD] = _g_local_ptr;

      // sync (ensuring the shared_array has been populated)
      upcxx::barrier();

      // copy remote pointers into _remote_ptrs cache, ensuring that future use
      // of remote pointers does not incur a global_ref.get()
      for ( int tid = 0; tid < NPROW * NPCOL; tid++ )
        // implicit type conversion in cast from shared_remote_ptrs[tid] to
        // global_ptr<T> (resolves remote global_ref<global_ptr<T> > reference
        // returned by shared_array's operator[])
        _remote_ptrs[tid] = shared_remote_ptrs[tid];

      // sync again (ensuring all threads have cached the global_ptrs before
      // shared_remote_ptrs goes out of scope)
      upcxx::barrier();
    }

    // initialize the update bins
    inline void
    init_bins()
    {
      // set up the bin objects
      for ( int tid = 0; tid < NPROW * NPCOL; tid++ )
#ifdef ENABLE_PROGRESS_THREAD
        _update_bins[tid].init( _remote_ptrs[tid], &_tq_mutex );
#else
        _update_bins[tid].init( _remote_ptrs[tid] );
#endif

      // set up random flushing order
      for ( int tid = 0; tid < NPROW * NPCOL; tid++ )
        _bin_flush_order[tid] = tid;
      permute( _bin_flush_order, NPROW * NPCOL );
    }

    // flush bins that are "full" (exceed the current threshold)
    inline void
    flush( int thresh = 0 )
    {
      // flush the bins
      for ( int b = 0; b < NPROW * NPCOL; b++ ) {
        int tid = _bin_flush_order[b];
        if ( _update_bins[tid].size() > thresh )
          _update_bins[tid].flush( &_e_update );
      }

#ifdef ENABLE_PROGRESS_THREAD
      if ( ! _progress_thread_running ) {
#endif
        // increment update counter
        _flush_counter += 1;

        // check whether we should pause to flush the task queue
        if ( thresh == 0 || _flush_counter == _progress_interval ) {
          // drain the task queue
          upcxx::drain();
          // reset the counter
          _flush_counter = 0;
        }
#ifdef ENABLE_PROGRESS_THREAD
      }
#endif
    }

    // determine lower bound on local storage for a given block-cyclic
    // distribution
    inline long
    roc( long m,   // matrix dimension
         long np,  // dimension of the process grid
         long ip,  // index in the process grid
         long mb ) // blocking factor
    {
      long nblocks, m_local;
      // number of full blocks to be distributed
      nblocks = m / mb;
      // set lower bound on required local dimension
      m_local = ( nblocks / np ) * mb;
      // edge cases ...
      if ( nblocks % np > ip )
        m_local += mb; // process ip receives another full block
      else if ( nblocks % np == ip )
        m_local += m % mb; // process ip _may_ receive a partial block
      return m_local;
    }

#ifdef ENABLE_PROGRESS_THREAD

    // begin progress thread execution (executes progress_helper())
    void
    progress_thread_start()
    {
      pthread_attr_t th_attr;
      progress_helper_args * args;

      // erroneous to call while thread is running
      assert( ! _progress_thread_running );

      // set up progress helper argument struct
      args = new progress_helper_args;
      args->tq_mutex = &_tq_mutex;
      args->progress_thread_stop = &_progress_thread_stop;

      // set thread as joinable
      pthread_attr_init( &th_attr );
      pthread_attr_setdetachstate( &th_attr, PTHREAD_CREATE_JOINABLE );

      // turn off stop flag
      _progress_thread_stop = false;

      // start the thread
      assert( pthread_create( &_progress_thread, &th_attr, progress_helper,
                              (void *)args ) == 0 );
      _progress_thread_running = true;
    }

    // signal to stop the progress thread and wait on it
    void
    progress_thread_stop()
    {
      // set the stop flag
      pthread_mutex_lock( &_tq_mutex );
      _progress_thread_stop = true;
      pthread_mutex_unlock( &_tq_mutex );

      // wait for thread to stop
      assert( pthread_join( _progress_thread, NULL ) == 0 );
      _progress_thread_running = false;
    }

#endif // ENABLE_PROGRESS_THREAD

#ifdef ENABLE_MPI_HELPERS

    inline MPI_Datatype
    get_mpi_base_type( float * )
    {
      return MPI_FLOAT;
    }

    inline MPI_Datatype
    get_mpi_base_type( double * )
    {
      return MPI_DOUBLE;
    }

    inline MPI_Datatype
    get_mpi_base_type()
    {
      return get_mpi_base_type( _local_ptr );
    }

#endif // ENABLE_MPI_HELPERS

#ifdef ENABLE_CONSISTENCY_CHECK

    inline void
    sum_updates( T *updates, T *summed_updates )
    {
      MPI_Datatype base_dtype = get_mpi_base_type();
      MPI_Allreduce( updates, summed_updates, _m * _n, base_dtype, MPI_SUM,
                     MPI_COMM_WORLD );
    }

    inline void
    consistency_check( T *updates )
    {
      long ncheck = 0;
      int mpi_init;
      const T rtol = 1e-8;
      T * summed_updates;

      // make sure MPI is already initialized
      assert( MPI_Initialized( &mpi_init ) == MPI_SUCCESS );
      assert( mpi_init );

      // sum the recorded updates across threads
      summed_updates = new T [_m * _n];
      sum_updates( updates, summed_updates );

      // ensure the locally-owned data is consistent with the record
      printf( "[%s] Thread %4i : Consistency check start ...\n", __func__,
              MYTHREAD );
      for ( long j = 0; j < _n; j++ )
        if ( ( j / NB ) % NPCOL == _mycol ) {
          long off_j = LLD * ( ( j / ( NB * NPCOL ) ) * NB + j % NB );
          for ( long i = 0; i < _m; i++ )
            if ( ( i / MB ) % NPROW == _myrow ) {
              long ij = off_j + ( i / ( MB * NPROW ) ) * MB + i % MB;
              T rres;
              if ( summed_updates[i + _m * j] == 0.0 )
                rres = 0.0;
              else
                rres = std::abs( ( summed_updates[i + _m * j] - _local_ptr[ij] )
                                 / summed_updates[i + _m * j] );
              assert( rres < rtol );
              ncheck += 1;
            }
        }
      delete [] summed_updates;
      printf( "[%s] Thread %4i : Consistency check PASSED for %li local"
              " entries\n", __func__, MYTHREAD, ncheck );
    }

#endif // ENABLE_CONSISTENCY_CHECK

   public:

    // ********************
    // ** public methods **
    // ********************

    /**
     * Initialize the \c ConvergentMatrix distributed matrix abstraction.
     * \param m Global leading dimension of the distributed matrix
     * \param n Global trailing dimension of the distributed matrix
     */
    ConvergentMatrix( long m, long n ) :
      _m(m), _n(n),
      _bin_flush_threshold(DEFAULT_BIN_FLUSH_THRESHOLD),
      _progress_interval(DEFAULT_PROGRESS_INTERVAL),
      _flush_counter(0)
    {
      // checks on matrix dimension
      assert( _m > 0 );
      assert( _n > 0 );

      // check on block-cyclic distribution
      assert( NPCOL * NPROW == THREADS );

      // setup block-cyclic distribution
      _myrow = MYTHREAD / NPROW;
      _mycol = MYTHREAD % NPCOL;

      // calculate minimum req'd local dimensions
      _m_local = roc( _m, NPROW, _myrow, MB );
      _n_local = roc( _n, NPCOL, _mycol, NB );

      // ensure local storage is of nonzero size
      assert( _m_local > 0 );
      assert( _n_local > 0 );

      // check minimum local leading dimension compatible with LLD
      assert( _m_local <= LLD );

      // initialize distributed storage
      init_arrays();

      // initialize update bins
      init_bins();

      // consistency check is off by default
#ifdef ENABLE_CONSISTENCY_CHECK
      _consistency_mode = false;
      _update_record = NULL;
#endif

      // initialize the task-queue mutex and progress thread state flag
#ifdef ENABLE_PROGRESS_THREAD
      _progress_thread_running = false;
      pthread_mutex_init( &_tq_mutex, NULL );
#endif

#ifdef ENABLE_UPDATE_TIMING
      _wt_init = omp_get_wtime();
#endif
    }

    /**
     * Destructor for the \c ConvergentMatrix distributed matrix abstraction.
     *
     * On destruction, will:
     * - Perform a final \c commit() (with consistency checks disabled),
     *   clearing remaining asynchronous tasks from the queue
     * - Free storage associated with update bins
     * - Free storage associated with the distributed matrix
     */
    ~ConvergentMatrix()
    {
      // run commit() one last time _without_ consistency checks
#ifdef ENABLE_CONSISTENCY_CHECK
      consistency_check_off();
#endif
      commit();

      // delete the GASNet-addressable local storage
      upcxx::deallocate<T>( upcxx::global_ptr<T>( _local_ptr ) );
    }

    /**
     * Get a raw pointer to the local distributed matrix storage (can be passed
     * to, for example, PBLAS routines).
     *
     * \b Note: The underlying storage _will_ be freed in the \c ConvergentMatrix
     * destructor - for a persistent copy, see \c get_local_data_copy().
     */
    inline T *
    get_local_data() const
    {
      return _local_ptr;
    }

    /**
     * Get a point to a _copy_ of the local distributed matrix storage (can be
     * passed to, for example, PBLAS routines).
     *
     * \b Note: The underlying storage will _not_ be freed in the
     * \c ConvergentMatrix destructor, in contrast to that from
     * \c get_local_data().
     */
    inline T *
    get_local_data_copy() const
    {
      T * copy_ptr = new T[LLD * _n_local];
      std::copy( _local_ptr, _local_ptr + LLD * _n_local, copy_ptr );
      return copy_ptr;
    }

    /**
     * Reset the distributed matrix (implicit barrier)
     *
     * Calls \c commit() (with consistency checks disabled), clearing remaining
     * asynchronous tasks from the queue, and then zeros the associated local
     * storage.
     *
     * \b Note: consistency checks must be manually re-enabled after calling
     * \c reset() (see \c consistency_check_on()).
     */
    inline void
    reset()
    {
      // disable consistency checks
#ifdef ENABLE_CONSISTENCY_CHECK
      consistency_check_off();
#endif

      // commit to complete any outstanding updates
      commit();

      // zero local storage
      std::fill( _local_ptr, _local_ptr + LLD * _n_local, 0 );

      // synchronize
      upcxx::barrier();
    }

    /**
     * Get the flush threshold (maximum bulk-update bin size before a bin is
     * flushed and applied to its target).
     */
    inline int
    bin_flush_threshold() const
    {
      return _bin_flush_threshold;
    }

    /**
     * Set the flush threshold (maximum bulk-update bin size before a bin is
     * flushed and applied to its target).
     * \param thresh The bin-size threshold
     */
    inline void
    bin_flush_threshold( int thresh )
    {
      _bin_flush_threshold = thresh;
    }

    /**
     * Get the progress interval, the number of bulk-update bin-flushes before
     * draining the local task queue.
     * Note that if there is a progress thread running, this drain operation is
     * not performed.
     */
    inline int
    progress_interval() const
    {
      return _progress_interval;
    }

    /**
     * Set the progress interval, the number of bulk-update bin-flushes before
     * draining the local task queue.
     * Note that if there is a progress thread running, this drain operation is
     * not performed.
     * \param interval The progress interval
     */
    inline void
    progress_interval( int interval )
    {
      _progress_interval = interval;
    }

    /**
     * Distributed matrix leading dimension
     */
    inline long
    m() const
    {
      return _m;
    }

    /**
     * Distributed matrix trailing dimension
     */
    inline long
    n() const
    {
      return _n;
    }

    /**
     * Process grid row index of this thread
     */
    inline long
    pgrid_row() const
    {
      return _myrow;
    }

    /**
     * Process grid column index of this thread
     */
    inline long
    pgrid_col() const
    {
      return _mycol;
    }

    /**
     * Minimum required leading dimension of local storage - must be less than
     * or equal to template parameter LLD
     */
    inline long
    m_local() const
    {
      return _m_local;
    }

    /**
     * Minimum required trailing dimension of local storage
     */
    inline long
    n_local() const
    {
      return _n_local;
    }

    /**
     * Remote random access (read only) to distributed matrix elements
     * \param ix Leading dimension index
     * \param jx Trailing dimension index
     */
    inline T
    operator()( long ix, long jx )
    {
      // infer thread id and linear index
      int tid = ( jx / NB ) % NPCOL + NPCOL * ( ( ix / MB ) % NPROW );
      long ij = LLD * ( ( jx / ( NB * NPCOL ) ) * NB + jx % NB ) +
                        ( ix / ( MB * NPROW ) ) * MB + ix % MB;
      // operator[ij] on global_ptr[tid] returns a global_ref to index ij on
      // target; implicit conversion (resolution) when returning type T
      return _remote_ptrs[tid][ij];
    }

    /**
     * Distributed matrix update: general case
     * \param Mat The update (strided) slice
     * \param ix Maps slice into distributed matrix (leading dimension)
     * \param jx Maps slice into distributed matrix (trailing dimension)
     */
    void
    update( LocalMatrix<T> *Mat, long *ix, long *jx )
    {
      // bin the local update
      for ( long j = 0; j < Mat->n(); j++ ) {
        int pcol = ( jx[j] / NB ) % NPCOL;
        long off_j = LLD * ( ( jx[j] / ( NB * NPCOL ) ) * NB + jx[j] % NB );
        for ( long i = 0; i < Mat->m(); i++ ) {
          int tid = pcol + NPCOL * ( ( ix[i] / MB ) % NPROW );
          long ij = off_j + ( ix[i] / ( MB * NPROW ) ) * MB + ix[i] % MB;
          _update_bins[tid].append( (*Mat)( i, j ), ij );
        }
      }

#ifdef ENABLE_CONSISTENCY_CHECK
      if ( _consistency_mode )
        for ( long j = 0; j < Mat->n(); j++ )
          for ( long i = 0; i < Mat->m(); i++ )
            (*_update_record)( ix[i], jx[j] ) += (*Mat)( i, j );
#endif

      // possibly flush bins
      flush( _bin_flush_threshold );
    }

    /**
     * Distributed matrix update: general _elemental_ case
     * \param elem The elemental update
     * \param ix Global index in distributed matrix (leading dimension)
     * \param jx Global index in distributed matrix (trailing dimension)
     */
    inline void
    update( T elem, long ix, long jx )
    {
      // bin the update
      int tid = ( jx / NB ) % NPCOL + NPCOL * ( ( ix / MB ) % NPROW );
      long ij = LLD * ( ( jx / ( NB * NPCOL ) ) * NB + jx % NB ) +
                        ( ix / ( MB * NPROW ) ) * MB + ix % MB;
      _update_bins[tid].append( elem, ij );

#ifdef ENABLE_CONSISTENCY_CHECK
      if ( _consistency_mode )
        (*_update_record)( ix, jx ) += elem;
#endif

      // possibly flush bins
      flush( _bin_flush_threshold );
    }

    /**
     * Distributed matrix update: symmetric case
     * \param Mat The update (strided) slice
     * \param ix Maps slice into distributed matrix (both dimensions)
     */
    void
    update( LocalMatrix<T> *Mat, long *ix )
    {
#ifdef ENABLE_UPDATE_TIMING
      double wt0, wt1;
#endif
#ifndef NOCHECK
      // must be square to be symmetric
      assert( Mat->m() == Mat->n() );
#endif

#ifdef ENABLE_UPDATE_TIMING
      wt0 = omp_get_wtime();
#endif
      // bin the local update
      for ( long j = 0; j < Mat->n(); j++ ) {
        int pcol = ( ix[j] / NB ) % NPCOL;
        long off_j = LLD * ( ( ix[j] / ( NB * NPCOL ) ) * NB + ix[j] % NB );
        for ( long i = 0; i < Mat->m(); i++ )
          if ( ix[i] <= ix[j] ) {
            int tid = pcol + NPCOL * ( ( ix[i] / MB ) % NPROW );
            long ij = off_j + ( ix[i] / ( MB * NPROW ) ) * MB + ix[i] % MB;
            _update_bins[tid].append( (*Mat)( i, j ), ij );
          }
      }
#ifdef ENABLE_UPDATE_TIMING
      wt1 = omp_get_wtime();
      printf( "[%s] %f binning time: %f s\n", __func__,
              wt1 - _wt_init, wt1 - wt0 );
#endif

#ifdef ENABLE_CONSISTENCY_CHECK
      if ( _consistency_mode )
        for ( long j = 0; j < Mat->n(); j++ )
          for ( long i = 0; i < Mat->m(); i++ )
            if ( ix[i] <= ix[j] )
              (*_update_record)( ix[i], ix[j] ) += (*Mat)( i, j );
#endif

      // possibly flush bins
#ifdef ENABLE_UPDATE_TIMING
      wt0 = omp_get_wtime();
#endif
      flush( _bin_flush_threshold );
#ifdef ENABLE_UPDATE_TIMING
      wt1 = omp_get_wtime();
      printf( "[%s] %f flush time: %f s\n", __func__,
              wt1 - _wt_init, wt1 - wt0 );
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
     * start.
     * Thus, always call \c commit() _before_ calling \c fill_lower(), and
     * again when you need to ensure the full updates have been applied.
     */
    void
    fill_lower()
    {
      for ( long j = 0; j < _n_local; j++ ) {
        long ix = ( j / NB ) * ( NPCOL * NB ) + _mycol * NB + j % NB;
        for ( long i = 0; i < _m_local; i++ ) {
          long jx = ( i / MB ) * ( NPROW * MB ) + _myrow * MB + i % MB;
          // use _transposed_ global indices to fill in the strict lower part
          if ( ix > jx ) {
            int tid = ( jx / NB ) % NPCOL + NPCOL * ( ( ix / MB ) % NPROW );
            long ij = LLD * ( ( jx / ( NB * NPCOL ) ) * NB + jx % NB ) +
                              ( ix / ( MB * NPROW ) ) * MB + ix % MB;
            _update_bins[tid].append( _local_ptr[i + j * LLD], ij );

#ifdef ENABLE_CONSISTENCY_CHECK
            if ( _consistency_mode )
              (*_update_record)( ix, jx ) += _local_ptr[i + j * LLD];
#endif

          } else {
            break; // nothing left to do in this column ...
          }
        }
      }

      // possibly flush bins
      flush( _bin_flush_threshold );
    }

    /**
     * Drain all update bins and wait on associated async tasks (implicit
     * barrier). If the consistency check is turned on, it will run after all
     * updates have been applied.
     *
     * \b Note: will stop the progress thread if it is running (see \c
     * use_progress_thread()).
     */
    inline void
    commit()
    {
      // stop the progress thread, if it has started
#ifdef ENABLE_PROGRESS_THREAD
      if ( _progress_thread_running )
        progress_thread_stop();
#endif

      // synchronize
      upcxx::barrier();

      // flush all non-empty bins (local task queue will be emptied)
      flush();

      // sync again (all locally-queued remote tasks have been dispatched)
      upcxx::barrier();

      // catch the last wave of tasks, if any
      upcxx::drain();

      // wait on remote tasks
      _e_update.wait();

      // done, sync on return
      upcxx::barrier();

      // if enabled, the consistency check should only occur after commit
#ifdef ENABLE_CONSISTENCY_CHECK
      if ( _consistency_mode )
        consistency_check( _update_record->data() );
#endif
    }

    // Optional functionality enabled at compile time ...

#ifdef ENABLE_PROGRESS_THREAD

    /**
     * Starts a progress thread for draining the task queue in the background.
     *
     * \b Importantly, the progress thread will only execute until the next
     * call to \c commit().
     * Thus, \c use_progress_thread() must be called for each commit epoch
     * separately if it is desired.
     * Further, the use of \c upcxx functions that touch the task queue during
     * progress thread execution is not advised and the resulting behavior is
     * \b undefined.
     *
     * \b Note: Requires compilation with \c ENABLE_PROGRESS_THREAD.
     */
    inline void
    use_progress_thread()
    {
      if ( ! _progress_thread_running )
        progress_thread_start();
    }

#endif // ENABLE_PROGRESS_THREAD

#ifdef ENABLE_CONSISTENCY_CHECK

    /**
     * Turn on consistency check mode.
     *
     * \b Note: MPI must be initialized in order for the consistency check to
     * run on calls to commit() (necessary for summation of the replicated
     * update records).
     *
     * \b Note: Requires compilation with \c ENABLE_CONSISTENCY_CHECK.
     */
    inline void
    consistency_check_on()
    {
      _consistency_mode = true;
      if ( _update_record == NULL )
        _update_record = new LocalMatrix<T>( _m, _n );
      (*_update_record) = (T) 0;
    }

    /**
     * Turn off consistency check mode
     *
     * \b Note: Requires compilation with \c ENABLE_CONSISTENCY_CHECK.
     */
    inline void
    consistency_check_off()
    {
      _consistency_mode = false;
      if ( _update_record != NULL )
        delete _update_record;
    }

#endif // ENABLE_CONSISTENCY_CHECK

#ifdef ENABLE_MPIIO_SUPPORT

    /**
     * Save the distributed matrix to disk via MPI-IO.
     * \param fname File name for matrix
     *
     * \b Note: No implicit \c commit() before matrix data is written - always
     * call \c commit() first.
     *
     * \b Note: Requires compilation with \c ENABLE_MPIIO_SUPPORT.
     */
    void
    save( const char *fname )
    {
      int mpi_init, mpi_rank, distmat_size, write_count;
      double wt_io, wt_io_max;
      MPI_Status status;
      MPI_Datatype distmat;
      MPI_File f_ata;

      // make sure we all get here
      upcxx::barrier();

      // make sure MPI is already initialized
      assert( MPI_Initialized( &mpi_init ) == MPI_SUCCESS );
      assert( mpi_init );

      // check process grid ordering
      MPI_Comm_rank( MPI_COMM_WORLD, &mpi_rank );
      assert( _myrow * NPCOL + _mycol == mpi_rank );

      // initialize distributed type
      int gsizes[]   = { (int)_m, (int)_n },
          distribs[] = { MPI_DISTRIBUTE_CYCLIC, MPI_DISTRIBUTE_CYCLIC },
          dargs[]    = { MB, NB },
          psizes[]   = { NPROW, NPCOL };
      MPI_Datatype base_dtype = get_mpi_base_type();
      MPI_Type_create_darray( NPCOL * NPROW, mpi_rank, 2,
                              gsizes, distribs, dargs, psizes,
                              MPI_ORDER_FORTRAN,
                              base_dtype,
                              &distmat );
      MPI_Type_commit( &distmat );

      // sanity check on check on distributed array data size
      MPI_Type_size( distmat, &distmat_size );
      assert( distmat_size / sizeof(T) == ( _m_local * _n_local ) );

      // open for writing
      MPI_File_open( MPI_COMM_WORLD, fname,
                     MPI_MODE_CREATE | MPI_MODE_WRONLY,
                     MPI_INFO_NULL, &f_ata );

      // set view w/ distmat
      MPI_File_set_view( f_ata, 0, base_dtype, distmat, "native", MPI_INFO_NULL );

      // compaction in place
      if ( _m_local < LLD )
        for ( long j = 1; j < _n_local; j++ )
          for ( long i = 0; i < _m_local; i++ )
            _local_ptr[i + j * _m_local] = _local_ptr[i + j * LLD];

      // write out local data
      wt_io = - MPI_Wtime();
      MPI_File_write_all( f_ata, _local_ptr, _m_local * _n_local, base_dtype, &status );
      wt_io = wt_io + MPI_Wtime();

      // close; report io time
      MPI_File_close( &f_ata );
      MPI_Reduce( &wt_io, &wt_io_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD );
      if ( mpi_rank == 0 )
        printf( "[%s] max time spent in matrix write: %.3f s\n", __func__,
                wt_io_max );

      // sanity check on data written
      MPI_Get_count( &status, base_dtype, &write_count );
      assert( write_count == ( _m_local * _n_local ) );

      // expansion in place
      if ( _m_local < LLD )
        for ( long j = _n_local - 1; j > 0; j-- )
          for ( long i = _m_local - 1; i >= 0; i-- )
            _local_ptr[i + j * LLD] = _local_ptr[i + j * _m_local];

      // sync post expansion (ops on distributed matrix elems can start again)
      upcxx::barrier();

      // free distributed type
      MPI_Type_free( &distmat );
    }

    /**
     * Load a distributed matrix to disk via MPI-IO.
     * \param fname File name for matrix
     *
     * \b Note: Replaces the current contents of the distributed storage array.
     *
     * \b Note: Requires compilation with \c ENABLE_MPIIO_SUPPORT.
     */
    void
    load( const char *fname )
    {
      int mpi_init, mpi_rank, distmat_size, read_count;
      double wt_io, wt_io_max;
      MPI_Status status;
      MPI_Datatype distmat;
      MPI_File f_ata;

      // make sure we all get here
      upcxx::barrier();

      // make sure MPI is already initialized
      assert( MPI_Initialized( &mpi_init ) == MPI_SUCCESS );
      assert( mpi_init );

      // check process grid ordering
      MPI_Comm_rank( MPI_COMM_WORLD, &mpi_rank );
      assert( _myrow * NPCOL + _mycol == mpi_rank );

      // initialize distributed type
      int gsizes[]   = { (int)_m, (int)_n },
          distribs[] = { MPI_DISTRIBUTE_CYCLIC, MPI_DISTRIBUTE_CYCLIC },
          dargs[]    = { MB, NB },
          psizes[]   = { NPROW, NPCOL };
      MPI_Datatype base_dtype = get_mpi_base_type();
      MPI_Type_create_darray( NPCOL * NPROW, mpi_rank, 2,
                              gsizes, distribs, dargs, psizes,
                              MPI_ORDER_FORTRAN,
                              base_dtype,
                              &distmat );
      MPI_Type_commit( &distmat );

      // sanity check on check on distributed array data size
      MPI_Type_size( distmat, &distmat_size );
      assert( distmat_size / sizeof(T) == ( _m_local * _n_local ) );

      // open read-only
      MPI_File_open( MPI_COMM_WORLD, fname,
                     MPI_MODE_RDONLY,
                     MPI_INFO_NULL, &f_ata );

      // set view w/ distmat
      MPI_File_set_view( f_ata, 0, base_dtype, distmat, "native", MPI_INFO_NULL );

      // read in local data
      wt_io = - MPI_Wtime();
      MPI_File_read_all( f_ata, _local_ptr, _m_local * _n_local, base_dtype, &status );
      wt_io = wt_io + MPI_Wtime();

      // close; report io time
      MPI_File_close( &f_ata );
      MPI_Reduce( &wt_io, &wt_io_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD );
      if ( mpi_rank == 0 )
        printf( "[%s] max time spent in matrix read: %.3f s\n", __func__,
                wt_io_max );

      // sanity check on data read
      MPI_Get_count( &status, base_dtype, &read_count );
      assert( read_count == ( _m_local * _n_local ) );

      // expansion in place
      if ( _m_local < LLD )
        for ( long j = _n_local - 1; j > 0; j-- )
          for ( long i = _m_local - 1; i >= 0; i-- )
            _local_ptr[i + j * LLD] = _local_ptr[i + j * _m_local];

      // sync post expansion (ops on distributed matrix elems can start again)
      upcxx::barrier();

      // free distributed type
      MPI_Type_free( &distmat );
    }

#endif // ENABLE_MPIIO_SUPPORT

  }; // end of ConvergentMatrix

} // end of namespace cm
