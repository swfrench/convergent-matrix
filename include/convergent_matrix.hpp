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
 *  - The ConvergentMatrix<T> abstraction accumulates updates to the global
 *    distributed matrix in bins (Bin<T>) for later asynchronous application.
 *
 *  - The Bin<T> object implements the binning concept in ConvergentMatrix,
 *    and handles "flushing" its contents by triggering remote asynchronous
 *    updates (update_task<T>).
 *
 * Once all updates have been applied, the matrix can be "frozen" and all bins
 * are flushed. Thereafter, each thread has its own PBLAS-compatible (using a
 * block-cyclic distribution and assuming a row-major process grid) portion of
 * the global matrix.
 */

#ifndef CONVERGENT_MATRIX_H
#define CONVERGENT_MATRIX_H

#include <iostream>
#include <vector>
#include <upcxx.h>

// overloaded definitions of gemm() and gemv() for double and float
#include "blas.hpp"

// block-cyclic distribution
//  ** the ConvergentMatrix abstraction is designed to be PBLAS compatible **
// TODO: move these to preprocessor defines?
#define MB 64     // local row-blocking factor
#define NB 64     // local col-blocking factor
#define NPROW 2   // number of rows in the process grid
#define NPCOL 2   // number of cols in the process grid
#define LLD 512   // local leading dimension of col-major storage

// default bin-size threshold (number of elems) before it is flushed
#define DEFAULT_BIN_FLUSH_THRESHOLD 1000

/**
 * Contains all classes associated with the convergent-matrix abstraction
 */
namespace convergent
{

  /**
   * A local dense matrix (supporting a _very_ limited set of blas ops)
   */
  template <typename T>
  class LocalMatrix
  {

    long _m, _n;
    long _ld;
    bool _trans;
    bool _alloc;
    T *_data;

  public:

    LocalMatrix( long m, long n )
    {
      assert( m > 0 );
      assert( n > 0 );
      _m = m;
      _n = n;
      _ld = m;
      _trans = false;
      _alloc = true;
      _data = new T[m * n];
    }

    LocalMatrix( long m, long n, T init )
    {
      assert( m > 0 );
      assert( n > 0 );
      _m = m;
      _n = n;
      _ld = m;
      _trans = false;
      _alloc = true;
      _data = new T[m * n];
      for ( long ij = 0; ij < m * n; ij++ )
        _data[ij] = init;
    }

    LocalMatrix( long m, long n, T *data )
    {
      assert( m > 0 );
      assert( n > 0 );
      _m = m;
      _n = n;
      _ld = m;
      _trans = false;
      _alloc = false;
      _data = data;
    }

    LocalMatrix( long m, long n, long ld, T *data )
    {
      assert( m > 0 );
      assert( n > 0 );
      _m = m;
      _n = n;
      _ld = ld;
      _trans = false;
      _alloc = false;
      _data = data;
    }

    ~LocalMatrix()
    {
      if ( _alloc )
        delete[] _data;
    }

    inline T&
    operator()( long i ) const
    {
#ifndef NOCHECK
      assert( _n == 1 );
      assert( i >= 0 );
      assert( i < m() );
#endif
      return _data[i];
    }

    inline T&
    operator()( long i, long j ) const
    {
#ifndef NOCHECK
      assert( i >= 0 );
      assert( i < m() );
      assert( j >= 0 );
      assert( j < n() );
#endif
      return _trans ? _data[j + i * _m] : _data[i + j * _m];
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

    inline bool
    is_col_vector()
    {
      return _trans ? ( _m == 1 ) : ( _n == 1 );
    }

    inline T *
    data() const
    {
      return _data;
    }

    inline T *
    col_data( long j ) const
    {
#ifndef NOCHECK
      // non-sensical under transpose view
      assert( ! _trans );
      assert( j < n() );
#endif
      return &_data[j * _m];
    }

    // free matrix data pointer in destructor, even though we did not allocate it
    inline void
    override_free()
    {
      _alloc = true;
    }

    LocalMatrix<T> *
    trans()
    {
      LocalMatrix<T> *TMat = new LocalMatrix<T>( _m, _n, _data );
      if ( ! _trans )
        TMat->_trans = true;
      return TMat;
    }

    LocalMatrix<T> *
    operator*( LocalMatrix<T> &BMat )
    {
      LocalMatrix<T> *CMat;
      T alpha = 1.0, beta = 0.0;

      // check on _view_ dimenions
      assert( n() == BMat.m() );

      // allocate storage for result
      CMat = new LocalMatrix<T>( m(), BMat.n() );

      if ( BMat._n == 1 )
        {
          // matrix-vector multiply
          char transa;
          int m, n, lda;
          int one = 1;
          m = _m; // rows of A
          n = _n; // cols of A
          lda = _ld;
          transa = _trans ? 'T' : 'N';
          gemv( &transa,
                &m, &n,
                &alpha,
                _data, &lda,
                BMat._data, &one,
                &beta,
                CMat->_data, &one );
        }
      else
        {
          // matrix-matrix multiply
          int m, n, k, lda, ldb, ldc;
          char transa, transb;
          m = _trans ? _n : _m;               // rows of op( A )
          n = BMat._trans? BMat._m : BMat._n; // cols of op( B )
          k = _trans ? _m : _n;               // cols of op( A )
          lda = _ld;
          ldb = BMat._ld;
          ldc = CMat->_ld;
          transa = _trans ? 'T' : 'N';
          transb = BMat._trans ? 'T' : 'N';
          gemm( &transa, &transb,
                &m, &n, &k,
                &alpha,
                _data, &lda,
                BMat._data, &ldb,
                &beta,
                CMat->_data, &ldc );
        }

      return CMat;
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
               upcxx::global_ptr<long> g_jx,
               upcxx::global_ptr<T>    g_data )
  {
    // local ptrs (for casts)
    long *p_ix, *p_jx;
    T *p_data, *p_my_data;

#ifdef DEBUGMSGS
    std::cout << "[" << __func__ << "] "
              << "Thread " << MYTHREAD << " "
              << "performing async update spawned by " << g_data.tid() << " "
              << "of size " << size
              << std::endl;
#endif

    // global pointers to local storage for remote copy
    upcxx::global_ptr<long> g_ix_local, g_jx_local;
    upcxx::global_ptr<T> g_data_local;

    // allocate local storage
    g_ix_local = upcxx::allocate<long>( MYTHREAD, size );
    g_jx_local = upcxx::allocate<long>( MYTHREAD, size );
    g_data_local = upcxx::allocate<T>( MYTHREAD, size );

    // copy to local
    upcxx::copy( g_ix, g_ix_local, size );
    upcxx::copy( g_jx, g_jx_local, size );
    upcxx::copy( g_data, g_data_local, size );

    // free remote storage
    upcxx::deallocate( g_ix );
    upcxx::deallocate( g_jx );
    upcxx::deallocate( g_data );

    // .. perform update ..

    // (1) cast to local ptrs
    p_ix = (long *) g_ix_local;
    p_jx = (long *) g_jx_local;
    p_data = (T *) g_data_local;
    p_my_data = (T *) g_my_data;

    // (2) update
    for ( long k = 0; k < size; k++ )
      {
        long ix =
          p_ix[k] / ( MB * NPROW ) * MB + p_ix[k] % MB +
          LLD * ( p_jx[k] / ( NB * NPCOL ) * NB + p_jx[k] % NB );
        p_my_data[ix] += p_data[k];
      }

    // (3) free local storage
    upcxx::deallocate( g_ix_local );
    upcxx::deallocate( g_jx_local );
    upcxx::deallocate( g_data_local );

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
    std::vector<long> _ix, _jx;
    std::vector<T> _data;

    inline void
    clear()
    {
      _ix.clear();
      _jx.clear();
      _data.clear();
    }

  public:

    Bin( upcxx::global_ptr<T> g_remote_data )
    {
      _remote_tid = g_remote_data.tid();
      _g_remote_data = g_remote_data;
    }

    inline void
    append( T data, long i, long j )
    {
      _ix.push_back( i );
      _jx.push_back( j );
      _data.push_back( data );
    }

    inline long
    size() const
    {
      return _ix.size();
    }

    // initiate remote async update using the current bin contents
    void
    flush()
    {
      // local ptrs (for casts)
      long *p_ix, *p_jx;
      T *p_data;

      // global ptrs to local storage for async remote copy
      upcxx::global_ptr<long> g_ix, g_jx;
      upcxx::global_ptr<T> g_data;

      // allocate local storage
      g_ix = upcxx::allocate<long>( MYTHREAD, _ix.size() );
      g_jx = upcxx::allocate<long>( MYTHREAD, _jx.size() );
      g_data = upcxx::allocate<T>( MYTHREAD, _data.size() );

      // fill local storage from accumulated updates
      p_ix = (long *)g_ix;
      p_jx = (long *)g_jx;
      p_data = (T *)g_data;
      for ( long i = 0; i < _ix.size(); i++ )
        {
          p_ix[i] = _ix[i];
          p_jx[i] = _jx[i];
          p_data[i] = _data[i];
        }

      // spawn the remote update (responsible for deallocating g_*)
      upcxx::async( _remote_tid )( update_task<float>,
                                   _data.size(),
                                   _g_remote_data,
                                   g_ix, g_jx, g_data );

      // clear internal (vector) storage
      clear();
    }

  }; // end of Bin


  /**
   * Convergent matrix abstraction
   */
  template <typename T>
  class ConvergentMatrix
  {

  private:

    long _m, _n;
    long _nbr, _nbc;
    int _bin_flush_threshold;
    bool _frozen;
    T *_local_ptr;
    upcxx::shared_array<upcxx::global_ptr<T> > _g_ptrs;
    std::vector<Bin<T> *> _bins;

    // flush bins that are "full" (exceed the current threshold)
    void
    bin_flush( int thresh )
    {
      for ( int tid = 0; tid < THREADS; tid++ )
        if ( _bins[tid]->size() > thresh )
          {
#ifdef DEBUG_MSGS
            std::cout << "[" << __func__ << "] "
                      << "flushing bin for tid " << tid
                      << std::endl;
#endif
            _bins[tid]->flush();
          }
    }

  public:

    ConvergentMatrix( long m, long n )
    {
      long ld_req;
      long myrow, mycol;

      // checks on matrix dimension
      assert( m > 0 );
      assert( n > 0 );

      // check on block-cyclic distribution
      assert( NPCOL * NPROW == THREADS );

      // store args
      _m = m;
      _n = n;

      // setup block-cyclic distribution
      myrow = MYTHREAD / NPROW;
      mycol = MYTHREAD % NPCOL;
      _nbr = _m / ( MB * NPROW ) + ( _m % ( MB * NPROW ) > myrow * MB ? 1 : 0 );
      _nbc = _n / ( NB * NPCOL ) + ( _n % ( NB * NPCOL ) > mycol * NB ? 1 : 0 );
#ifdef DEBUG_MSGS
      std::cout << "[" << __func__ << "] "
                << "Thread " << MYTHREAD
                << " [ " << myrow
                << " / " << mycol
                << " ] " << _nbr << " x " << _nbc << " local blocks"
                << std::endl;
#endif

      // allocate local storage, exchange global ptrs ...

      // (1) check minimum local leading dimension
      ld_req = _nbr * MB;
      assert( ld_req <= LLD );

      // (2) initialize shared_array of global pointers
      _g_ptrs.init( THREADS );

      // (3) allocate storage, cast to global_ptr, write to _g_ptrs
      _local_ptr = new T [LLD * _nbc * NB];
      _g_ptrs[MYTHREAD] = (upcxx::global_ptr<T>)(_local_ptr);
      upcxx::barrier();

      // set flush threashold for bins
      _bin_flush_threshold = DEFAULT_BIN_FLUSH_THRESHOLD;

      // set up bins
      for ( int tid = 0; tid < THREADS; tid++ )
        _bins.push_back( new Bin<T>( _g_ptrs[tid] ) );

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
    bin_flush_threshold()
    {
      return _bin_flush_threshold;
    }

    inline void
    bin_flush_threshold( int thresh )
    {
      _bin_flush_threshold = thresh;
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
          for ( long i = 0; i < Mat->m(); i++ )
            {
              int tid = pcol + NPCOL * ( ( ix[i] / MB ) % NPROW );
              _bins[tid]->append( (*Mat)( i, j ), ix[i], jx[j] );
            }
        }

      // possibly flush bins
      bin_flush( _bin_flush_threshold );
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
          for ( long i = j; i < Mat->m(); i++ )
            {
              int tid = pcol + NPCOL * ( ( ix[i] / MB ) % NPROW );
              _bins[tid]->append( (*Mat)( i, j ), ix[i], ix[j] );
            }
        }

      // possibly flush bins
      bin_flush( _bin_flush_threshold );
    }

    // drain _all_ of the bins
    void
    freeze()
    {
      bin_flush( 0 );
      upcxx::wait();
      upcxx::barrier();
      _frozen = true;
    }

  }; // end of ConvergentMatrix

} // end of namespace convergent

#endif
