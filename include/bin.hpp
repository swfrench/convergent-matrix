#pragma once

#include <cassert>
#include <vector>

#ifdef ENABLE_PROGRESS_THREAD
#include <pthread.h>
#endif

#ifdef FLUSH_ALLOC_RETRY
#include <algorithm> // std::min
#include <unistd.h>  // usleep
#ifndef RETRY_MIN_INTERVAL
#define RETRY_MIN_INTERVAL 10       // ms
#endif
#ifndef RETRY_MAX_INTERVAL
#define RETRY_MAX_INTERVAL 1000     // ms
#endif
#ifndef RETRY_INTERVAL_FACTOR
#define RETRY_INTERVAL_FACTOR 2
#endif
#ifndef RETRY_MAX_ITER
#define RETRY_MAX_ITER 1000
#endif
#endif /* FLUSH_ALLOC_RETRY */

#include <upcxx.h>

// retry remote allocation statement A with bounded exponential backoff
#ifdef FLUSH_ALLOC_RETRY
#ifdef FLUSH_WARN_ON_RETRY
#define FLUSH_WARN_RETRY if ( iter > 0 ) \
  printf( "Warning: Thread %4i : ALLOC_WRAP required %li attempts\n", \
          MYTHREAD, iter );
#else
#define FLUSH_WARN_RETRY /* noop */
#endif /* FLUSH_WARN_RETRY */
#define ALLOC_WRAP( A ) \
  do { \
    long iter = 0; \
    useconds_t t = RETRY_MIN_INTERVAL; \
    while ( ( A ).raw_ptr() == NULL && iter++ < RETRY_MAX_ITER ) { \
      usleep( 1000 * t ); \
      t = std::min( t * RETRY_INTERVAL_FACTOR, \
                    (useconds_t) RETRY_MAX_INTERVAL ); \
    } \
    FLUSH_WARN_RETRY \
  } while(0)
#else
// identity
#define ALLOC_WRAP( A ) A
#endif /* FLUSH_ALLOC_RETRY */

namespace cm
{

  /// @cond INTERNAL_DOCS

  /**
   * A task that performs remote updates (spawned by Bin<T>)
   * \param g_my_data Reference to local storage on the target
   * \param g_ix Reference to update indexing (already transferred target)
   * \param g_data Reference to update data (already transferred target)
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
    assert( g_my_data.where() == MYTHREAD );
    assert( g_ix.where()      == MYTHREAD );
    assert( g_data.where()    == MYTHREAD );
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

    bool _init;                           // whether fields intialized
    int _remote_tid;                      // thread id of target
    upcxx::global_ptr<T> _g_remote_data;  // global_ptr _local_ to target
    std::vector<long> _ix;                // linear indexing for target
    std::vector<T> _data;                 // update data for target
#ifdef ENABLE_PROGRESS_THREAD
    pthread_mutex_t *_tq_mutex;           // mutex protecting task-queue ops
#endif

    inline void
    clear()
    {
      _ix.clear();
      _data.clear();
    }

   public:

    /**
     * Create an uninitialized Bin object
     */
    Bin() :
      _init(false)
    {}

#ifdef ENABLE_PROGRESS_THREAD
    /**
     * Create the Bin object associated with a given target
     * \param g_remote_data global_ptr<T> reference to the target-local storage
     * \param tq_mutex pthread mutex protecting the task queue and associated
     * GASNet operations
     */
    Bin( upcxx::global_ptr<T> g_remote_data, pthread_mutex_t * tq_mutex ) :
      _init(true),
      _remote_tid(g_remote_data.where()),
      _g_remote_data(g_remote_data),
      _tq_mutex(tq_mutex)
    {}

    /**
     * Initialize the previously empty Bin object for a given target
     * \param g_remote_data global_ptr<T> reference to the target-local storage
     * \param tq_mutex pthread mutex protecting the task queue and associated
     * GASNet operations
     *
     * \b Note: It is erroneous to call this on a previously initialized Bin.
     */
    inline void
    init( upcxx::global_ptr<T> g_remote_data, pthread_mutex_t * tq_mutex )
    {
      assert( ! _init );
      _remote_tid = g_remote_data.where();
      _g_remote_data = g_remote_data;
      _tq_mutex = tq_mutex;
      _init = true;
    }
#else
    /**
     * Create the Bin object associated with a given target
     * \param g_remote_data global_ptr<T> reference to the target-local storage
     */
    Bin( upcxx::global_ptr<T> g_remote_data ) :
      _init(true),
      _remote_tid(g_remote_data.where()),
      _g_remote_data(g_remote_data)
    {}

    /**
     * Initialize the previously empty Bin object for a given target
     * \param g_remote_data global_ptr<T> reference to the target-local storage
     *
     * \b Note: It is erroneous to call this on a previously initialized Bin.
     */
    inline void
    init( upcxx::global_ptr<T> g_remote_data )
    {
      assert( ! _init );
      _remote_tid = g_remote_data.where();
      _g_remote_data = g_remote_data;
      _init = true;
    }
#endif

    /**
     * Current size of the bin (number of update elems not yet applied)
     */
    inline long
    size() const
    {
      return _ix.size();
    }

    /**
     * Add an elemental update to this bin
     * \param data Elemental update (r.h.s. of +=)
     * \param ij Linear index (on the target) of element to be updated
     */
    inline void
    append( T data, long ij )
    {
      _ix.push_back( ij );
      _data.push_back( data );
    }

    /**
     * "Flush" this bin by initiate remote async update using the current bin
     * contents.
     * \param e upcxx::event pointer to which the async task will be registered
     */
    void
    flush( upcxx::event *e )
    {
#ifndef NOCHECK
      assert( _init );
#endif

      // global ptrs to local storage for async remote copy
      upcxx::global_ptr<long> g_ix;
      upcxx::global_ptr<T> g_data;

#ifdef ENABLE_PROGRESS_THREAD
      pthread_mutex_lock( _tq_mutex );
#endif

      // allocate remote storage
      ALLOC_WRAP( g_ix = upcxx::allocate<long>( _remote_tid, _ix.size() ) );
#ifdef NOCHECK
      assert( g_ix.raw_ptr() != NULL );
#endif
      ALLOC_WRAP( g_data = upcxx::allocate<T>( _remote_tid, _data.size() ) );
#ifdef NOCHECK
      assert( g_data.raw_ptr() != NULL );
#endif

      // copy to remote
      upcxx::copy( (upcxx::global_ptr<long>) _ix.data(), g_ix, _ix.size() );
      upcxx::copy( (upcxx::global_ptr<T>) _data.data(), g_data, _data.size() );

      // spawn the remote update (responsible for deallocating g_ix, g_data)
      upcxx::async( _remote_tid, e )( update_task<T>,
                                      _data.size(),
                                      _g_remote_data,
                                      g_ix, g_data );

#ifdef ENABLE_PROGRESS_THREAD
      pthread_mutex_unlock( _tq_mutex );
#endif

      // clear internal (vector) storage
      clear();
    }

  }; // end of Bin

  /// @endcond

}
