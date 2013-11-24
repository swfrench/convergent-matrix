/**
 * A "convergent" distributed dense matrix data structure
 *
 * Key components:
 *
 *  - The LocalMatrix<T> object, which simply represents a local matrix and
 *    supports a few common BLAS operations; these are how the updates are
 *    represented locally, along with index arrays which map into the global
 *    index space.
 *
 *  - The ConvergentMatrix<T,NPROW,NPCOL,MB,NB,LLD> abstraction accumulates
 *    updates to the global distributed matrix in bins (Bin<T>) for later
 *    asynchronous application.
 *
 *  - The Bin<T> object implements the binning concept in ConvergentMatrix,
 *    and handles "flushing" its contents by triggering remote asynchronous
 *    updates (update_task<T>).
 *
 * Once all updates have been applied, the matrix can be "frozen" and all bins
 * are flushed. Thereafter, each thread has its own PBLAS-compatible portion of
 * the global matrix, consistent with the block-cyclic distribution defined by
 * the template parameters of ConvergentMatrix and assuming a row-major order
 * of threads in the process grid.
 */

#pragma once

#include <iostream>
#include <vector>
#include <cassert>
#include <upcxx.h>

// overloaded definitions of gemm() and gemv() for double and float
#include "blas.hpp"

// default bin-size threshold (number of elems) before it is flushed
#define DEFAULT_BIN_FLUSH_THRESHOLD 10000

// default number of update() to trigger progress() and drain (local) task queue
#define DEFAULT_PROGRESS_INTERVAL 10

/**
 * Contains classes associated with the convergent-matrix abstraction
 */
namespace cm
{

  /**
   * A local dense matrix abstraction supporting a limited set of level 2 and 3
   * BLAS operations and element-wise arithmetic
   */
  template <typename T>
  class LocalMatrix
  {

   private:

    long _m, _n, _ld;
    bool _alloc;
    bool _trans;
    T * _data;

   public:

    LocalMatrix() :
      _m(0), _n(0), _ld(0), _alloc(false), _trans(false), _data(NULL)
    {}

    LocalMatrix( long m, long n ) :
      _m(m), _n(n), _ld(m), _alloc(true), _trans(false), _data(new T[m * n])
    {}

    LocalMatrix( long m, long n, T * data ) :
      _m(m), _n(n), _ld(m), _alloc(false), _trans(false), _data(data)
    {}

    LocalMatrix( long m, long n, long ld ) :
      _m(m), _n(n), _ld(ld), _alloc(true), _trans(false), _data(new T[ld * n])
    {}

    LocalMatrix( long m, long n, long ld, T * data ) :
      _m(m), _n(n), _ld(ld), _alloc(false), _trans(false), _data(data)
    {}

    ~LocalMatrix()
    {
      if ( _alloc )
        delete [] _data;
    }

    inline void
    override_free()
    {
      _alloc = true;
    }

    inline long
    m() const
    {
      return _trans ? _n : _m;
    }

    inline long
    n() const
    {
      return _trans ? _m : _n;
    }

    LocalMatrix<T> *
    trans()
    {
      LocalMatrix<T> * C = new LocalMatrix<T>( _m, _n, _ld, _data );
      C->_trans = ! _trans;
      return C;
    }

    inline T&
    operator()( long i, long j ) const
    {
#ifndef NOCHECK
      assert( i >= 0 );
      assert( j >= 0 );
      assert( i < m() );
      assert( j < n() );
#endif
      return _trans ? _data[j + i * _ld] : _data[i + _ld * j];
    }

    inline T&
    operator()( long i ) const
    {
#ifndef NOCHECK
      assert( ( m() == 1 &&   _trans ) ||
              ( n() == 1 && ! _trans ) );
#endif
      return _data[i];
    }

    LocalMatrix<T> &
    operator=( T val )
    {
      for ( long j = 0; j < n(); j++ )
        for ( long i = 0; i < m(); i++ )
          (*this)( i, j ) = val;
      return *this;
    }

    LocalMatrix<T> *
    operator+( LocalMatrix<T> &B ) const
    {
      LocalMatrix<T> *C = new LocalMatrix<T>( _m, _n, _ld );
      if ( _trans )
        C->trans();
#ifndef NOCHECK
      assert( m() == B.m() );
      assert( n() == B.n() );
#endif
      for ( long j = 0; j < n(); j++ )
        for ( long i = 0; i < m(); i++ )
          (*C)( i, j ) = (*this)( i, j ) + B( i, j );
      return C;
    }

    LocalMatrix<T> *
    operator-( LocalMatrix<T> &B ) const
    {
      LocalMatrix<T> *C = new LocalMatrix<T>( _m, _n, _ld );
      if ( _trans )
        C->trans();
#ifndef NOCHECK
      assert( m() == B.m() );
      assert( n() == B.n() );
#endif
      for ( long j = 0; j < n(); j++ )
        for ( long i = 0; i < m(); i++ )
          (*C)( i, j ) = (*this)( i, j ) - B( i, j );
      return C;
    }

    LocalMatrix<T> &
    operator+=( LocalMatrix<T> &B )
    {
#ifndef NOCHECK
      assert( m() == B.m() );
      assert( n() == B.n() );
#endif
      for ( long j = 0; j < n(); j++ )
        for ( long i = 0; i < m(); i++ )
          (*this)( i, j ) += B( i, j );
      return *this;
    }

    LocalMatrix<T> &
    operator-=( LocalMatrix<T> &B )
    {
#ifndef NOCHECK
      assert( m() == B.m() );
      assert( n() == B.n() );
#endif
      for ( long j = 0; j < n(); j++ )
        for ( long i = 0; i < m(); i++ )
          (*this)( i, j ) -= B( i, j );
      return *this;
    }

    LocalMatrix<T> *
    operator*( LocalMatrix<T> &B ) const
    {
      LocalMatrix<T> *C;
      T alpha = 1.0, beta = 0.0;

      // check on _view_ dimenions
      assert( n() == B.m() );

      C = new LocalMatrix<T>( m(), B.n() );

      if ( B.n() == 1 )
        {
#ifndef NOCHECK
          assert( ! B._trans ||
                  ( B._trans && B._ld == 1 ) );
#endif
          char transa;
          int m, n, lda;
          int one = 1;
          m = _m; // true rows of A
          n = _n; // true cols of A
          lda = _ld;
          transa = _trans ? 'T' : 'N';
          gemv( &transa,
                &m, &n,
                &alpha,
                _data, &lda,
                B._data, &one,
                &beta,
                C->_data, &one );
        }
      else
        {
          int m, n, k, lda, ldb, ldc;
          char transa, transb;
          m = this->m(); // rows of op( A )
          n = B.n();     // cols of op( B )
          k = this->n(); // cols of op( A )
          lda = _ld;
          ldb = B._ld;
          ldc = C->_ld;
          transa = _trans ? 'T' : 'N';
          transb = B._trans ? 'T' : 'N';
          gemm( &transa, &transb,
                &m, &n, &k,
                &alpha,
                _data, &lda,
                B._data, &ldb,
                &beta,
                C->_data, &ldc );
        }

      return C;
    }

  }; // end of LocalMatrix


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
    T *_local_ptr;
    std::vector<Bin<T> *> _bins;
    upcxx::event _e;
    upcxx::shared_array<upcxx::global_ptr<T> > _g_ptrs;

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

      // allocate local storage, exchange global ptrs ...

      // (1) check minimum local leading dimension
      ld_req = _nbr * MB;
      assert( ld_req <= LLD );

      // (2) initialize shared_array of global pointers
      _g_ptrs.init( THREADS );

      // (3) allocate and zero storage; cast to global_ptr and write to _g_ptrs
      _local_ptr = new T [LLD * _nbc * NB]();
      _g_ptrs[MYTHREAD] = (upcxx::global_ptr<T>)(_local_ptr);
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
    }

    ~ConvergentMatrix()
    {
      // it is assumed that the user will free _local_ptr
      for ( int tid = 0; tid < THREADS; tid++ )
        delete _bins[tid];
    }

    // returns view of local block-cyclic storage as a LocalMatrix
    // ** no copy is performed - may continue to mutate if called before freeze() **
    inline LocalMatrix<T> *
    as_local_matrix()
    {
      return new LocalMatrix<T>( _nbr * MB, _nbc * NB, LLD, _local_ptr );
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

    inline void
    pgrid_row() const
    {
      return _myrow;
    }

    inline void
    pgrid_col() const
    {
      return _mycol;
    }

    // general case
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
          long off_j = LLD * ( jx[j] / ( NB * NPCOL ) * NB + jx[j] % NB );
          for ( long i = 0; i < Mat->m(); i++ )
            {
              int tid = pcol + NPCOL * ( ( ix[i] / MB ) % NPROW );
              long ij = off_j + ix[i] / ( MB * NPROW ) * MB + ix[i] % MB;
              _bins[tid]->append( (*Mat)( i, j ), ij );
            }
        }

      // possibly flush bins
      flush( _bin_flush_threshold );
    }

    // symmetric case
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
          long off_j = LLD * ( ix[j] / ( NB * NPCOL ) * NB + ix[j] % NB );
          for ( long i = j; i < Mat->m(); i++ )
            {
              int tid = pcol + NPCOL * ( ( ix[i] / MB ) % NPROW );
              long ij = off_j + ix[i] / ( MB * NPROW ) * MB + ix[i] % MB;
              _bins[tid]->append( (*Mat)( i, j ), ij );
            }
        }

      // possibly flush bins
      flush( _bin_flush_threshold );
    }

    // drain the bins, stop accepting updates
    inline void
    freeze()
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
    }

  }; // end of ConvergentMatrix

} // end of namespace cm
