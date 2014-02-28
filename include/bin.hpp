#pragma once

#include <cassert>
#include <vector>
#include <upcxx.h>

namespace cm
{

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

    /**
     * Initialize the Bin object for a given target
     * \param g_remote_data global_ptr<T> reference to the target-local storage
     */
    Bin( upcxx::global_ptr<T> g_remote_data ) :
      _remote_tid(g_remote_data.tid()), _g_remote_data(g_remote_data)
    {}

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

}
