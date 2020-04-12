#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <upcxx/upcxx.hpp>

#include "include/bin.hpp"

#define ASSERT(want, got, msg)                                                \
  do {                                                                        \
    if ((got) != (want)) {                                                    \
      std::cerr << __FILE__ << ":" << __LINE__ << "] " << msg                 \
                << " : expected equality want=" << (want) << " got=" << (got) \
                << std::endl;                                                 \
      exit(1);                                                                \
    }                                                                         \
  } while (false)

namespace test {

// Each process writes its own rank at index `rank` in other processes' local
// portion of the dist_object.
void singleUpdate() {
  const int rank = upcxx::rank_me();
  const int num_ranks = upcxx::rank_n();

  upcxx::dist_object<upcxx::global_ptr<int>> g_data(
      upcxx::new_array<int>(num_ranks));

  std::vector<cm::Bin<int>> bins(num_ranks);
  for (int i = 0; i < num_ranks; ++i) bins[i].init(g_data.fetch(i).wait());

  for (int i = 0; i < num_ranks; ++i) bins[i].append(rank, rank);

  upcxx::promise<> p_op;
  for (auto &bin : bins) bin.flush(&p_op);

  p_op.finalize().wait();

  upcxx::barrier();

  const int *local = g_data->local();
  for (int i = 0; i < num_ranks; ++i)
    ASSERT(i, local[i], "Verification failed");
}

// TODO: Add more complex test case.

}  // namespace test

int main(int argc, char *argv[]) {
  upcxx::init();
  test::singleUpdate();
  upcxx::finalize();
  return 0;
}
