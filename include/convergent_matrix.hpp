/**
 * A "convergent" distributed dense matrix data structure
 *
 * Key components:
 *
 *  - The ConvergentMatrix<T,NPROW,NPCOL,MB,NB,LLD> abstraction accumulates
 *    updates to the global distributed matrix in bins (Bin<T>) for later
 *    asynchronous application.
 *
 *  - The Bin<T> object implements the binning concept in ConvergentMatrix,
 *    and handles "flushing" its contents by triggering remote asynchronous
 *    updates (update_task<T>).
 *
 * Updates are represented as LocalMatrix<T> objects (see local_matrix.hpp),
 * along with indexing arrays which map into the global distributed index
 * space.
 *
 * Once all updates have been applied, the matrix can be "frozen" and all bins
 * are flushed. Thereafter, each thread has its own PBLAS-compatible portion of
 * the global matrix, consistent with the block-cyclic distribution defined by
 * the template parameters of ConvergentMatrix and assuming a row-major order
 * of threads in the process grid.
 */

#pragma once

#if ( defined(DEBUG_MSGS)       || \
      defined(ASYNC_DEBUG_MSGS) || \
      defined(TEST_CONSISTENCY) )
#include <iostream>
#endif
#include <vector>
#include <cassert>

#include <upcxx.h>

#ifdef TEST_CONSISTENCY
#include <cmath>
#include <mpi.h>
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

#if defined(DEBUG_MSGS) || defined(ASYNC_DEBUG_MSGS)
    assert( g_my_data.tid() == MYTHREAD );
    assert( g_ix.tid() == MYTHREAD );
    assert( g_data.tid() == MYTHREAD );
    std::cout << "[" << __func__ << "] "
              << "Thread " << MYTHREAD << " "
              << "performing async update of size " << size
              << std::endl;
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
   */
  template<typename T>
  class Bin
  {

   private:

    int _remote_tid;
    upcxx::global_ptr<T> _g_remote_data;
    std::vector<long> _ix;
    std::vector<T> _data;

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

#if defined(DEBUG_MSGS) || defined(ASYNC_DEBUG_MSGS)
      std::cout << "[" << __func__ << "] "
                << "Thread " << MYTHREAD << " "
                << "called async() for update of size " << _data.size() << " "
                << "on thread " << _remote_tid
                << std::endl;
#endif

      // clear internal (vector) storage
      clear();
    }

  }; // end of Bin


  /**
   * Convergent matrix abstraction
   */
  template <typename T,              // matrix type
            long NPROW, long NPCOL,  // pblas process grid
            long MB, long NB,        // pblas local blocking factors
            long LLD>                // pblas local leading dim
  class ConvergentMatrix
  {

   private:

    long _m, _n;
    long _nbr, _nbc;
    long _myrow, _mycol;
    int _flush_counter;
    int _progress_interval;
    int _bin_flush_threshold;
    bool _frozen;
    std::vector<Bin<T> *> _bins;
    T *_local_ptr;
    upcxx::global_ptr<T> _g_local_ptr;
    upcxx::event _e;
    upcxx::shared_array<upcxx::global_ptr<T> > _g_ptrs;
#ifdef TEST_CONSISTENCY
    LocalMatrix<T> * _record;
#endif

    // drain the entire task queue
    inline void
    drain_task_queue()
    {
      upcxx::drain();
    }

    // flush bins that are "full" (exceed the current threshold)
    inline void
    flush( int thresh = 0 )
    {
      // flush the bins
      for ( int tid = 0; tid < THREADS; tid++ )
        if ( _bins[tid]->size() > thresh )
          {
#ifdef DEBUG_MSGS
            std::cout << "[" << __func__ << "] "
                      << "Thread " << MYTHREAD << " "
                      << "flushing bin for tid " << tid
                      << std::endl;
#endif
            _bins[tid]->flush( &_e );
#ifdef DEBUG_MSGS
            std::cout << "[" << __func__ << "] "
                      << "Thread " << MYTHREAD << " "
                      << "returned from flush() in bin for tid " << tid
                      << std::endl;
#endif
          }

      // increment update counter
      _flush_counter += 1;

      // check whether we should pause to flush the task queue
      if ( thresh == 0 || _flush_counter == _progress_interval )
        {
          // drain the task queue
          drain_task_queue();
          // reset the counter
          _flush_counter = 0;
        }
    }

   public:

    ConvergentMatrix( long m, long n ) :
      _m(m), _n(n)
    {
      long ld_req;
#ifdef TEST_CONSISTENCY
      int mpi_init;
#endif

      // checks on matrix dimension
      assert( _m > 0 );
      assert( _n > 0 );

      // check on block-cyclic distribution
      assert( NPCOL * NPROW == THREADS );

      // setup block-cyclic distribution
      _myrow = MYTHREAD / NPROW;
      _mycol = MYTHREAD % NPCOL;
      _nbr = _m / ( MB * NPROW ) + ( _m % ( MB * NPROW ) > _myrow * MB ? 1 : 0 );
      _nbc = _n / ( NB * NPCOL ) + ( _n % ( NB * NPCOL ) > _mycol * NB ? 1 : 0 );
#ifdef DEBUG_MSGS
      std::cout << "[" << __func__ << "] "
                << "Thread " << MYTHREAD
                << " [ " << _myrow
                << " / " << _mycol
                << " ] " << _nbr << " x " << _nbc << " local blocks"
                << std::endl;
#endif

      // ensure local storage is of nonzero size
      assert( _nbr > 0 );
      assert( _nbc > 0 );

      // allocate local storage, exchange global ptrs ...

      // (1) check minimum local leading dimension
      ld_req = _nbr * MB;
      assert( ld_req <= LLD );

      // (2) initialize shared_array of global pointers
      _g_ptrs.init( THREADS );

      // (3) allocate and zero storage; write to _g_ptrs
      _g_local_ptr = upcxx::allocate<T>( MYTHREAD, LLD * _nbc * NB );
      _local_ptr = (T *) _g_local_ptr;
      for ( long ij = 0; ij < LLD * _nbc * NB; ij++ )
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

      // set _frozen to false, and we're open for business
      _frozen = false;

#ifdef TEST_CONSISTENCY
      std::cout << "[" << __func__ << "] "
                << "Thread " << MYTHREAD
                << " Initialized in consistency test mode"
                << std::endl;
      _record = new LocalMatrix<T>( _m, _n );
      assert( MPI_Initialized( &mpi_init ) == MPI_SUCCESS );
      assert( mpi_init );
#endif
    }

    ~ConvergentMatrix()
    {
      for ( int tid = 0; tid < THREADS; tid++ )
        delete _bins[tid];
      upcxx::deallocate<T>( _g_local_ptr );
    }

    inline T *
    get_local_data()
    {
      assert( _frozen );
      return _local_ptr;
    }

    inline void
    reset()
    {
      // must already be in frozen state (all updates committed)
      assert( _frozen );
      // zero local storage
      for ( long ij = 0; ij < LLD * _nbc * NB; ij++ )
        _local_ptr[ij] = (T) 0;
      // unfreeze
      _frozen = false;
#ifdef TEST_CONSISTENCY
      // reset consistency check ground truth as well
      _record = new LocalMatrix<T>( _m, _n );
#endif
      // must be called by all threads
      upcxx::barrier();
    }

    inline int
    bin_flush_threshold() const
    {
      return _bin_flush_threshold;
    }

    inline void
    bin_flush_threshold( int thresh )
    {
      _bin_flush_threshold = thresh;
    }

    inline int
    progress_interval() const
    {
      return _progress_interval;
    }

    inline void
    progress_interval( int interval )
    {
      _progress_interval = interval;
    }

    inline long
    pgrid_row() const
    {
      return _myrow;
    }

    inline long
    pgrid_col() const
    {
      return _mycol;
    }

    inline long
    m() const
    {
      return _m;
    }

    inline long
    n() const
    {
      return _n;
    }

    inline long
    pgrid_nrow() const
    {
      return NPROW;
    }

    inline long
    pgrid_ncol() const
    {
      return NPCOL;
    }

    inline long
    mb() const
    {
      return MB;
    }

    inline long
    nb() const
    {
      return NB;
    }

    inline long
    lld() const
    {
      return LLD;
    }

    // remote random access (read only)
    inline T
    operator()( long ix, long jx )
    {
#ifndef NOCHECK
      assert( _frozen );
#endif
      int tid = ( jx / NB ) % NPCOL + NPCOL * ( ( ix / MB ) % NPROW );
      long ij = LLD * ( ( jx / ( NB * NPCOL ) ) * NB + jx % NB ) +
                        ( ix / ( MB * NPROW ) ) * MB + ix % MB;
      // temporary hack: long index into global_ptr not currently supported
      return _g_ptrs[tid].get() [(int)ij];
    }

    // distributed matrix update: general case
    void
    update( LocalMatrix<T> *Mat, long *ix, long *jx )
    {
#ifndef NOCHECK
      assert( ! _frozen );
#endif
      // bin the local update
      for ( long j = 0; j < Mat->n(); j++ )
        {
          int pcol = ( jx[j] / NB ) % NPCOL;
          long off_j = LLD * ( ( jx[j] / ( NB * NPCOL ) ) * NB + jx[j] % NB );
          for ( long i = 0; i < Mat->m(); i++ )
            {
              int tid = pcol + NPCOL * ( ( ix[i] / MB ) % NPROW );
              long ij = off_j + ( ix[i] / ( MB * NPROW ) ) * MB + ix[i] % MB;
              _bins[tid]->append( (*Mat)( i, j ), ij );
#ifdef TEST_CONSISTENCY
              (*_record)( ix[i], jx[j] ) += (*Mat)( i, j );
#endif
            }
        }

      // possibly flush bins
      flush( _bin_flush_threshold );
    }

    // distributed matrix update: symmetric case
    void
    update( LocalMatrix<T> *Mat, long *ix )
    {
#ifndef NOCHECK
      assert( ! _frozen );
      assert( Mat->m() == Mat->n() );
#endif
      // bin the local update
      for ( long j = 0; j < Mat->n(); j++ )
        {
          int pcol = ( ix[j] / NB ) % NPCOL;
          long off_j = LLD * ( ( ix[j] / ( NB * NPCOL ) ) * NB + ix[j] % NB );
          for ( long i = 0; i < Mat->m(); i++ )
            if ( ix[i] <= ix[j] )
              {
                int tid = pcol + NPCOL * ( ( ix[i] / MB ) % NPROW );
                long ij = off_j + ( ix[i] / ( MB * NPROW ) ) * MB + ix[i] % MB;
                _bins[tid]->append( (*Mat)( i, j ), ij );
#ifdef TEST_CONSISTENCY
                (*_record)( ix[i], ix[j] ) += (*Mat)( i, j );
#endif
              }
        }

      // possibly flush bins
      flush( _bin_flush_threshold );
    }

#ifdef TEST_CONSISTENCY

    inline void
    sum_updates( float *updates )
    {
      MPI_Allreduce( MPI_IN_PLACE, updates, _m * _n, MPI_FLOAT, MPI_SUM,
                     MPI_COMM_WORLD );
    }

    inline void
    sum_updates( double *updates )
    {
      MPI_Allreduce( MPI_IN_PLACE, updates, _m * _n, MPI_DOUBLE, MPI_SUM,
                     MPI_COMM_WORLD );
    }

    inline void
    consistency_check( T *updates )
    {
      long ncheck = 0;
      const T rtol = 1e-8;
      // sum the recorded updates across threads
      sum_updates( updates );
      // ensure the locally-owned data is consistent with the record
      std::cout << "[" << __func__ << "] "
                << "Thread " << MYTHREAD
                << " Consistency check start ..."
                << std::endl;
      for ( long j = 0; j < _n; j++ )
        {
          if ( ( j / NB ) % NPCOL == _mycol )
            {
              long off_j = LLD * ( ( j / ( NB * NPCOL ) ) * NB + j % NB );
              for ( long i = 0; i < _m; i++ )
                if ( ( i / MB ) % NPROW == _myrow )
                  {
                    long ij = off_j + ( i / ( MB * NPROW ) ) * MB + i % MB;
                    T rres;
                    if ( updates[i + _m * j] == 0.0 )
                      rres = 0.0;
                    else
                      rres = std::abs( ( updates[i + _m * j] - _local_ptr[ij] )
                                       / updates[i + _m * j] );
                    assert( rres < rtol );
                    ncheck += 1;
                  }
            }
        }
      std::cout << "[" << __func__ << "] "
                << "Thread " << MYTHREAD
                << " Consistency check PASSED for "
                << ncheck << " local entries"
                << std::endl;
    }

#endif // TEST_CONSISTENCY

    // drain the bins, stop accepting updates
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
      drain_task_queue();
      // wait on remote tasks
      _e.wait();
      // done, sync on return
      upcxx::barrier();
      // stop accepting updates
      _frozen = true;
      // if enabled, the consistency check should only occur after commit
#ifdef TEST_CONSISTENCY
      consistency_check( _record->data() );
      delete _record;
#endif
    }

  }; // end of ConvergentMatrix

} // end of namespace cm
