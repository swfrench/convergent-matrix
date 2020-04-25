#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include <upcxx/upcxx.hpp>

#include "include/bin.hpp"

#define LOG \
  std::cerr << __FILE__ << ":" << __LINE__ << " @ " << upcxx::rank_me() << "] "

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
  for (int i = 0; i < num_ranks; ++i) {
    if (local[i] == i) continue;
    LOG << "verification failed at index " << i << " want: " << i
        << " got: " << local[i] << std::endl;
    exit(1);
  }
}

// Each process increments a randomly selected index at a randomly selected
// target rank.
void randomMultiUpdate() {
  constexpr int niter = 10, nupdate = 1000, nlocal = 1000;

  const int rank = upcxx::rank_me();
  const int num_ranks = upcxx::rank_n();

  std::vector<int> local_mirror(nlocal * num_ranks, 0);

  upcxx::dist_object<upcxx::global_ptr<int>> g_data(
      upcxx::new_array<int>(nlocal));

  const int *local = g_data->local();

  std::vector<cm::Bin<int>> bins(num_ranks);
  for (int i = 0; i < num_ranks; ++i) bins[i].init(g_data.fetch(i).wait());

  std::random_device rdev;
  std::mt19937 rgen(rdev());
  std::uniform_int_distribution<long> ix_dist(0, nlocal - 1);
  std::uniform_int_distribution<long> rank_dist(0, num_ranks - 1);

  for (int iter = 0; iter < niter; ++iter) {
    for (int n = 0; n < nupdate; ++n) {
      const int target = rank_dist(rgen), i = ix_dist(rgen);
      bins[target].append(1, i);
      local_mirror[target * nlocal + i] += 1;
    }

    upcxx::promise<> p_op;
    for (auto &bin : bins) bin.flush(&p_op);

    p_op.finalize().wait();

    upcxx::barrier();

    std::vector<int> sum(local_mirror.size(), 0);
    upcxx::reduce_all(local_mirror.data(), sum.data(), local_mirror.size(),
                      upcxx::op_fast_add)
        .wait();

    for (int i = 0; i < nlocal; ++i) {
      if (local[i] == sum[nlocal * rank + i]) continue;
      LOG << "verification failed at index " << i << " want: " << i
          << " got: " << local[i] << std::endl;
      exit(1);
    }

    // Wait for verification to complete across all processes before starting
    // the next round (lest we potentially observe remotely injected updates).
    upcxx::barrier();
  }
}

}  // namespace test

int main(int argc, char *argv[]) {
  upcxx::init();
  test::singleUpdate();
  test::randomMultiUpdate();
  upcxx::finalize();
  return 0;
}
