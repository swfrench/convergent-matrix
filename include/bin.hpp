#pragma once

#include <cassert>
#include <vector>
#ifdef ENABLE_PROGRESS_THREAD
#include <pthread.h>
#endif
#include <upcxx.h>

namespace cm
{

  /// @cond INTERNAL_DOCS

  // a tuple containing the elemental update and its (remote) linear index
  template<typename T>
  struct elem
  {
    T element;
    long index;
    elem( T element_, long index_ ) :
      element(element_), index(index_) {};
  };

  /**
   * A task that performs remote updates (spawned by Bin<T>)
   * \param g_my_data Reference to local storage on the target
   * \param g_data Reference to update data (already transferred target)
   */
  template <typename T>
  void
  update_task( long size,
               upcxx::global_ptr<T>        g_my_data,
               upcxx::global_ptr<elem<T> > g_data )
  {
    // local ptrs (for casts)
    elem<T> *p_data;
    T *p_my_data;

#ifndef NOCHECK
    assert( g_my_data.tid() == MYTHREAD );
    assert( g_data.tid()    == MYTHREAD );
#endif

    // cast to local ptrs
    p_data = (elem<T> *) g_data;
    p_my_data = (T *) g_my_data;

    // update
    for ( long k = 0; k < size; k++ )
      p_my_data[p_data[k].index] += p_data[k].element;

    // free local (but remotely allocated) storage
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
    std::vector<elem<T> > _data;          // update data for target
#ifdef ENABLE_PROGRESS_THREAD
    pthread_mutex_t *_tq_mutex;           // mutex protecting task-queue ops
#endif

    inline void
    clear()
    {
      _data.clear();
    }

   public:

#ifdef ENABLE_PROGRESS_THREAD
    /**
     * Initialize the Bin object for a given target
     * \param g_remote_data global_ptr<T> reference to the target-local storage
     * \param tq_mutex pthread mutex protecting the task queue and associated
     * GASNet operations
     */
    Bin( upcxx::global_ptr<T> g_remote_data, pthread_mutex_t * tq_mutex ) :
      _remote_tid(g_remote_data.tid()), _g_remote_data(g_remote_data), _tq_mutex(tq_mutex)
    {}
#else
    /**
     * Initialize the Bin object for a given target
     * \param g_remote_data global_ptr<T> reference to the target-local storage
     */
    Bin( upcxx::global_ptr<T> g_remote_data ) :
      _remote_tid(g_remote_data.tid()), _g_remote_data(g_remote_data)
    {}
#endif

    /**
     * Current size of the bin (number of update elems not yet applied)
     */
    inline long
    size() const
    {
      return _data.size();
    }

    /**
     * Add an elemental update to this bin
     * \param data Elemental update (r.h.s. of +=)
     * \param ij Linear index (on the target) of element to be updated
     */
    inline void
    append( T data, long ij )
    {
      _data.push_back( elem<T>( data, ij ) );
    }

    /**
     * "Flush" this bin by initiate remote async update using the current bin
     * contents.
     * \param e upcxx::event pointer to which the async task will be registered
     */
    void
    flush( upcxx::event *e )
    {
      // global ptrs to local storage for async remote copy
      upcxx::global_ptr<elem<T> > g_data;

#ifdef ENABLE_PROGRESS_THREAD
      pthread_mutex_lock( _tq_mutex );
#endif

      // allocate remote storage
      g_data = upcxx::allocate<elem<T> >( _remote_tid, _data.size() );

      // copy to remote
      upcxx::copy( (upcxx::global_ptr<elem<T> >) _data.data(), g_data, _data.size() );

      // spawn the remote update (responsible for deallocating g_ix, g_data)
      upcxx::async( _remote_tid, e )( update_task<T>,
                                      _data.size(),
                                      _g_remote_data,
                                      g_data );

#ifdef ENABLE_PROGRESS_THREAD
      pthread_mutex_unlock( _tq_mutex );
#endif

      // clear internal (vector) storage
      clear();
    }

  }; // end of Bin

  /// @endcond

}
