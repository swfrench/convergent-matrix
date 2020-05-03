#pragma once

#include <cassert>
#include <vector>

#include <upcxx/upcxx.hpp>

namespace cm {

/// @cond INTERNAL_DOCS

/**
 * Applies remote updates via RPC (spawned by Bin<T>)
 * \param g_my_data Reference to local storage on the target
 * \param g_ix Update indexing (already transferred target)
 * \param g_data Update data (already transferred target)
 */
template <typename T>
void update_task(long size, upcxx::global_ptr<T> g_my_data,
                 upcxx::global_ptr<long> g_ix, upcxx::global_ptr<T> g_data) {
#ifndef NOCHECK
  assert(g_my_data.is_local());
  assert(g_ix.is_local());
  assert(g_data.is_local());
#endif
  long *buff_ix = g_ix.local();
  T *buff_data = g_data.local();
  T *my_data = g_my_data.local();
  for (long k = 0; k < size; k++) my_data[buff_ix[k]] += buff_data[k];
  // clean up transfer buffers
  upcxx::delete_array(g_ix);
  upcxx::delete_array(g_data);
}

/**
 * Implements the binning / remote application for a single participating
 * process. Not very efficient in terms of space (e.g. duplicate updates to the
 * same remote data elements are not merged), in exchange for simplicity.
 * \tparam T Matrix data type (e.g. float)
 *
 * This class is not thread safe, however it is thread compatible.
 */
template <typename T>
class Bin {
 private:
  bool _init;                           // whether fields intialized
  int _remote_rank;                     // rank of target
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
        _remote_rank(g_remote_data.where()),
        _g_remote_data(g_remote_data) {}

  /**
   * Initialize the previously empty Bin object for a given target
   * \param g_remote_data global_ptr<T> reference to the target-local storage
   *
   * \b Note: It is erroneous to call this on a previously initialized Bin.
   */
  void init(upcxx::global_ptr<T> g_remote_data) {
    assert(!_init);
    _remote_rank = g_remote_data.where();
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
   * "Flush" this bin by initiating a remote update with the current bin
   * contents.
   * \param p_op upcxx::promise<> pointer to which RPC remote operation
   * completion will be registered
   */
  void flush(upcxx::promise<> *p_op) {
#ifndef NOCHECK
    assert(_init);
#endif

    // allocate remote buffers for update data
    auto buffers =
        upcxx::rpc(
            _remote_rank,
            [](long size)
                -> std::pair<upcxx::global_ptr<long>, upcxx::global_ptr<T>> {
              return {upcxx::new_array<long>(size), upcxx::new_array<T>(size)};
            },
            size())
            .wait();

    // transfer update data to the target
    upcxx::promise<> p_put;
    upcxx::rput(_ix.data(), buffers.first, size(),
                upcxx::operation_cx::as_promise(p_put));
    upcxx::rput(_data.data(), buffers.second, size(),
                upcxx::operation_cx::as_promise(p_put));
    p_put.finalize().wait();  // now safe to clear()

    upcxx::rpc(_remote_rank, upcxx::operation_cx::as_promise(*p_op),
               update_task<T>, size(), _g_remote_data, buffers.first,
               buffers.second);

    clear();
  }

};  // end of Bin

/// @endcond

}  // namespace cm
