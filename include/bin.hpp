#pragma once

#include <cassert>
#include <vector>

#include <upcxx/upcxx.hpp>

namespace cm {

/// @cond INTERNAL_DOCS

/**
 * A task that performs remote updates (spawned by Bin<T>)
 * \param g_my_data Reference to local storage on the target
 * \param ix View to update indexing (already transferred target)
 * \param data View to update data (already transferred target)
 */
template <typename T>
void update_task(long size, upcxx::global_ptr<T> g_my_data,
                 upcxx::view<long> ix, upcxx::view<T> data) {
#ifndef NOCHECK
  assert(g_my_data.is_local());
#endif
  T *my_data = g_my_data.local();
  for (long k = 0; k < size; k++) my_data[ix[k]] += data[k];
}

/**
 * Implements the binning / remote application for a single thread
 * Not very efficient in terms of space for the moment, in exchange for
 * simplicity.
 * \tparam T Matrix data type (e.g. float)
 *
 * This class is not thread safe, however it is thread compatible.
 */
template <typename T>
class Bin {
 private:
  bool _init;                           // whether fields intialized
  int _remote_tid;                      // thread id of target
  upcxx::global_ptr<T> _g_remote_data;  // global_ptr _local_ to target
  std::vector<long> _ix;                // linear indexing for target
  std::vector<T> _data;                 // update data for target

  void clear() {
    _ix.clear();
    _data.clear();
  }

 public:
  /**
   * Create an uninitialized Bin object
   */
  Bin() : _init(false) {}

  /**
   * Create the Bin object associated with a given target
   * \param g_remote_data global_ptr<T> reference to the target-local storage
   */
  Bin(upcxx::global_ptr<T> g_remote_data)
      : _init(true),
        _remote_tid(g_remote_data.where()),
        _g_remote_data(g_remote_data) {}

  /**
   * Initialize the previously empty Bin object for a given target
   * \param g_remote_data global_ptr<T> reference to the target-local storage
   *
   * \b Note: It is erroneous to call this on a previously initialized Bin.
   */
  void init(upcxx::global_ptr<T> g_remote_data) {
    assert(!_init);
    _remote_tid = g_remote_data.where();
    _g_remote_data = g_remote_data;
    _init = true;
  }

  /**
   * Current size of the bin (number of update elems not yet applied)
   */
  long size() const { return _ix.size(); }

  /**
   * Add an elemental update to this bin
   * \param data Elemental update (r.h.s. of +=)
   * \param ij Linear index (on the target) of element to be updated
   */
  void append(T data, long ij) {
    _ix.push_back(ij);
    _data.push_back(data);
  }

  /**
   * "Flush" this bin by initiate remote async update using the current bin
   * contents.
   * \param p_op upcxx::promise<> pointer to which RPC remote operation
   * completion will be registered
   */
  void flush(upcxx::promise<> *p_op) {
#ifndef NOCHECK
    assert(_init);
#endif

    // track completion of source-side serialization to render clear() safe:
    auto cxs = upcxx::source_cx::as_buffered() |
               upcxx::operation_cx::as_promise(*p_op);
    upcxx::rpc(_remote_tid, cxs, update_task<T>, _data.size(), _g_remote_data,
               upcxx::make_view(_ix), upcxx::make_view(_data));

    // now safe to call clear()
    clear();
  }

};  // end of Bin

/// @endcond

}  // namespace cm
