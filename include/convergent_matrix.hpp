/**
 * A "convergent" distributed dense matrix data structure
 *
 * Key components:
 *  - The ConvergentMatrix<T,NPROW,NPCOL,MB,NB,LLD> abstraction accumulates
 *    updates to the global distributed matrix in bins (Bin<T>) for later
 *    asynchronous application.
 *  - The Bin<T> object implements the binning concept in ConvergentMatrix,
 *    and handles "flushing" its contents by triggering remote asynchronous
 *    updates (update_task<T>).
 *
 * Updates are represented as LocalMatrix<T> objects (see local_matrix.hpp),
 * along with indexing arrays which map into the global distributed index
 * space.
 *
 * Once a series of updates have been applied, the matrix can be "committed"
 * (see ConvergentMatrix::commit()) and all bins are flushed.
 * Thereafter, each thread has its own PBLAS-compatible portion of the global
 * matrix, consistent with the block-cyclic distribution defined by the
 * template parameters of ConvergentMatrix and assuming a row-major order of
 * threads in the process grid.
 */

#pragma once

#include <vector>
#include <cstdio>
#include <cassert>
#ifdef ENABLE_CONSISTENCY_CHECK
#include <cmath>
#endif

#include <upcxx.h>

#if ( defined(ENABLE_CONSISTENCY_CHECK) || \
      defined(ENABLE_MPIIO_SUPPORT) )
#include <mpi.h>
#define ENABLE_MPI_HELPERS
#endif

// LocalMatrix<T>
#include "local_matrix.hpp"

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

  /**
   * A task that performs remote updates (spawned by Bin<T>)
   */
  template <typename T>
  void
  update_task( long size,
               upcxx::global_ptr<T>    g_my_data,
               upcxx::global_ptr<long> g_ix,
               upcxx::global_ptr<T>    g_data )
  {
    // local ptrs (for casts)
    long *p_ix;
    T *p_data, *p_my_data;

#ifndef NOCHECK
    assert( g_my_data.tid() == MYTHREAD );
    assert( g_ix.tid()      == MYTHREAD );
    assert( g_data.tid()    == MYTHREAD );
#endif

    // cast to local ptrs
    p_ix = (long *) g_ix;
    p_data = (T *) g_data;
    p_my_data = (T *) g_my_data;

    // update
    for ( long k = 0; k < size; k++ )
      p_my_data[p_ix[k]] += p_data[k];

    // free local (but remotely allocated) storage
    upcxx::deallocate( g_ix );
    upcxx::deallocate( g_data );

  } // end of update_task


  /**
   * Implements the binning / remote application for a single thread
   * Not very efficient in terms of space for the moment, in exchange for
   * simplicity.
   * \tparam T Matrix data type (e.g. float)
   */
  template<typename T>
  class Bin
  {

   private:

    int _remote_tid;                      // thread id of target
    upcxx::global_ptr<T> _g_remote_data;  // global_ptr _local_ to target
    std::vector<long> _ix;                // linear indexing for target
    std::vector<T> _data;                 // update data for target

    inline void
    clear()
    {
      _ix.clear();
      _data.clear();
    }

   public:

    Bin( upcxx::global_ptr<T> g_remote_data ) :
      _remote_tid(g_remote_data.tid()), _g_remote_data(g_remote_data)
    {}

    inline void
    append( T data, long ij )
    {
      _ix.push_back( ij );
      _data.push_back( data );
    }

    inline long
    size() const
    {
      return _ix.size();
    }

    // initiate remote async update using the current bin contents
    void
    flush( upcxx::event *e )
    {
      // global ptrs to local storage for async remote copy
      upcxx::global_ptr<long> g_ix;
      upcxx::global_ptr<T> g_data;

      // allocate remote storage
      g_ix = upcxx::allocate<long>( _remote_tid, _ix.size() );
      g_data = upcxx::allocate<T>( _remote_tid, _data.size() );

      // copy to remote
      upcxx::copy( (upcxx::global_ptr<long>) _ix.data(), g_ix, _ix.size() );
      upcxx::copy( (upcxx::global_ptr<T>) _data.data(), g_data, _data.size() );

      // spawn the remote update (responsible for deallocating g_ix, g_data)
      upcxx::async( _remote_tid, e )( update_task<T>,
                                      _data.size(),
                                      _g_remote_data,
                                      g_ix, g_data );

      // clear internal (vector) storage
      clear();
    }

  }; // end of Bin


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

    long _m, _n;
    long _myrow, _mycol;
    long _m_local, _n_local;
    int _flush_counter;
    int _progress_interval;
    int _bin_flush_threshold;
    std::vector<Bin<T> *> _bins;
    T *_local_ptr;
    upcxx::global_ptr<T> _g_local_ptr;
    upcxx::event _e;
    upcxx::shared_array<upcxx::global_ptr<T> > _g_ptrs;
#ifdef ENABLE_CONSISTENCY_CHECK
    bool _consistency_mode;
    LocalMatrix<T> * _update_record;
#endif

    // flush bins that are "full" (exceed the current threshold)
    inline void
    flush( int thresh = 0 )
    {
      // flush the bins
      for ( int tid = 0; tid < THREADS; tid++ )
        if ( _bins[tid]->size() > thresh )
          _bins[tid]->flush( &_e );

      // increment update counter
      _flush_counter += 1;

      // check whether we should pause to flush the task queue
      if ( thresh == 0 || _flush_counter == _progress_interval ) {
        // drain the task queue
        upcxx::drain();
        // reset the counter
        _flush_counter = 0;
      }
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

#ifdef ENABLE_MPI_HELPERS

    inline int
    get_mpi_base_type( float *x )
    {
      return MPI_FLOAT;
    }

    inline int
    get_mpi_base_type( double *x )
    {
      return MPI_DOUBLE;
    }

    inline
    int get_mpi_base_type()
    {
      return get_mpi_base_type( _local_ptr );
    }

#endif

#ifdef ENABLE_CONSISTENCY_CHECK

    inline void
    sum_updates( T *updates, T *summed_updates )
    {
      int base_dtype = get_mpi_base_type();
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

    /**
     * The ConvergentMatrix distributed matrix abstraction.
     * \param m Global leading dimension
     * \param n Global trailing dimension
     */
    ConvergentMatrix( long m, long n ) :
      _m(m), _n(n)
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

      // allocate local storage, exchange global ptrs ...

      // (1) check minimum local leading dimension
      assert( _m_local <= LLD );

      // (2) initialize shared_array of global pointers
      _g_ptrs.init( THREADS );

      // (3) allocate and zero storage; write to _g_ptrs
      _g_local_ptr = upcxx::allocate<T>( MYTHREAD, LLD * _n_local );
      _local_ptr = (T *) _g_local_ptr;
      for ( long ij = 0; ij < LLD * _n_local; ij++ )
        _local_ptr[ij] = (T) 0;
      _g_ptrs[MYTHREAD] = _g_local_ptr;
      upcxx::barrier();

      // set flush threashold for bins
      _bin_flush_threshold = DEFAULT_BIN_FLUSH_THRESHOLD;

      // set up bins
      for ( int tid = 0; tid < THREADS; tid++ )
        _bins.push_back( new Bin<T>( _g_ptrs[tid] ) );

      // init the progress() interval to its default value
      _progress_interval = DEFAULT_PROGRESS_INTERVAL;

      // init counter for initiating calls to progress()
      _flush_counter = 0;

      // consistency check is off by default
#ifdef ENABLE_CONSISTENCY_CHECK
      _consistency_mode = false;
      _update_record = NULL;
#endif
    }

    ~ConvergentMatrix()
    {
      for ( int tid = 0; tid < THREADS; tid++ )
        delete _bins[tid];
      upcxx::deallocate<T>( _g_local_ptr );
    }

    /**
     * Get a raw pointer to the local distributed matrix storage (can be passed
     * to, for example, PBLAS routines).
     */
    inline T *
    get_local_data()
    {
      return _local_ptr;
    }

    /**
     * Reset the distributed matrix (implicit barrier). Zeros the associated
     * local storage (as well as the update record if consistency checks are
     * turned on).
     */
    inline void
    reset()
    {
      // zero local storage
      for ( long ij = 0; ij < LLD * _n_local; ij++ )
        _local_ptr[ij] = (T) 0;
#ifdef ENABLE_CONSISTENCY_CHECK
      // reset consistency check ground truth as well
      if ( _consistency_mode )
        (*_update_record) = (T) 0;
#endif
      // must be called by all threads
      upcxx::barrier();
    }

    /**
     * get the flush threshold (maximum bulk-update bin size before a bin is
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
     * Get the progress interval (number of calls to update() before draining
     * the local task queue)
     */
    inline int
    progress_interval() const
    {
      return _progress_interval;
    }

    /**
     * Set the progress interval (number of calls to update() before draining
     * the local task queue)
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
      int tid = ( jx / NB ) % NPCOL + NPCOL * ( ( ix / MB ) % NPROW );
      long ij = LLD * ( ( jx / ( NB * NPCOL ) ) * NB + jx % NB ) +
                        ( ix / ( MB * NPROW ) ) * MB + ix % MB;
      // temporary hack: long index into global_ptr not currently supported
      return _g_ptrs[tid].get() [(int)ij];
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
          _bins[tid]->append( (*Mat)( i, j ), ij );
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
     * Distributed matrix update: symmetric case
     * \param Mat The update (strided) slice
     * \param ix Maps slice into distributed matrix (both dimensions)
     */
    void
    update( LocalMatrix<T> *Mat, long *ix )
    {
#ifndef NOCHECK
      // must be square to be symmetric
      assert( Mat->m() == Mat->n() );
#endif
      // bin the local update
      for ( long j = 0; j < Mat->n(); j++ ) {
        int pcol = ( ix[j] / NB ) % NPCOL;
        long off_j = LLD * ( ( ix[j] / ( NB * NPCOL ) ) * NB + ix[j] % NB );
        for ( long i = 0; i < Mat->m(); i++ )
          if ( ix[i] <= ix[j] ) {
            int tid = pcol + NPCOL * ( ( ix[i] / MB ) % NPROW );
            long ij = off_j + ( ix[i] / ( MB * NPROW ) ) * MB + ix[i] % MB;
            _bins[tid]->append( (*Mat)( i, j ), ij );
          }
      }
#ifdef ENABLE_CONSISTENCY_CHECK
      if ( _consistency_mode )
        for ( long j = 0; j < Mat->n(); j++ )
          for ( long i = 0; i < Mat->m(); i++ )
            if ( ix[i] <= ix[j] )
              (*_update_record)( ix[i], ix[j] ) += (*Mat)( i, j );
#endif

      // possibly flush bins
      flush( _bin_flush_threshold );
    }

    /**
     * Drain all update bins and wait on associated async tasks (implicit
     * barrier). If the consistency check is turned on, it will run after all
     * updates have been applied.
     */
    inline void
    commit()
    {
      // synchronize
      upcxx::barrier();

      // flush all non-empty bins (local task queue will be emptied)
      flush();

      // sync again (all locally-queued remote tasks have been dispatched)
      upcxx::barrier();

      // catch the last wave of tasks, if any
      upcxx::drain();

      // wait on remote tasks
      _e.wait();

      // done, sync on return
      upcxx::barrier();

      // if enabled, the consistency check should only occur after commit
#ifdef ENABLE_CONSISTENCY_CHECK
      if ( _consistency_mode )
        consistency_check( _update_record->data() );
#endif
    }

#ifdef ENABLE_CONSISTENCY_CHECK

    /**
     * Turn on consistency check mode (requires compilation with
     * ENABLE_CONSISTENCY_CHECK).
     * NOTE: MPI must be initialized in order for the consistency check to run
     * on calls to commit() (necessary for summation of the replicated update
     * records).
     */
    inline void
    consistency_check_on()
    {
      _consistency_mode = true;
      if ( _update_record == NULL )
        _update_record = new LocalMatrix<T>( _m, _n );
      (*_update_record) = (T) 0;
      printf( "[%s] Thread %4i : Consistency check mode ON (recording ...)\n",
              __func__, MYTHREAD );
    }

    /**
     * Turn off consistency check mode (requires compilation with
     * ENABLE_CONSISTENCY_CHECK).
     */
    inline void
    consistency_check_off()
    {
      _consistency_mode = false;
      if ( _update_record != NULL )
        delete _update_record;
      printf( "[%s] Thread %4i : Consistency check mode OFF\n",
              __func__, MYTHREAD );
    }

#endif // ENABLE_CONSISTENCY_CHECK

#ifdef ENABLE_MPIIO_SUPPORT

    /**
     * Save the distributed matrix to disk via MPI-IO (requres compilation with
     * ENABLE_MPIIO_SUPPORT). No implicit commit() before matrix data is
     * written - always call commit() first.
     * \param fname File name for matrix
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
          psizes[]   = { NPROW, NPCOL },
          base_dtype = get_mpi_base_type();
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

      // free distributed type
      MPI_Type_free( &distmat );
    }

#endif // ENABLE_MPIIO_SUPPORT

  }; // end of ConvergentMatrix

} // end of namespace cm
