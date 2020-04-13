#include <cassert>
#include <cstdlib>
#include <iostream>
#include <random>

#include <upcxx/upcxx.hpp>

#include "include/convergent_matrix.hpp"
#include "include/local_matrix.hpp"

#define ASSERT(want, got, msg)                                                \
  do {                                                                        \
    if ((got) != (want)) {                                                    \
      std::cerr << __FILE__ << ":" << __LINE__ << "] " << msg                 \
                << " : expected equality want=" << (want) << " got=" << (got) \
                << std::endl;                                                 \
      exit(1);                                                                \
    }                                                                         \
  } while (false)

#define MB 64
#define NB 64
#define NPROW 2
#define NPCOL 2

namespace test {

// TODO: Better coverage of multi-epoch updates (i.e. multiple calls to commit).

void randomElementUpdates() {
  constexpr int niter = 1000;

  cm::LocalMatrix<int> local_mirror(2000, 1000);
  cm::ConvergentMatrix<int, NPROW, NPCOL, MB, NB,
                       /*LLD=*/1024>
      dist_mat(2000, 1000);

  std::random_device rdev;
  std::mt19937 rgen(rdev());
  std::uniform_int_distribution<long> row_dist(0, dist_mat.m() - 1);
  std::uniform_int_distribution<long> col_dist(0, dist_mat.n() - 1);

  for (int n = 0; n < niter; ++n) {
    const long i = row_dist(rgen), j = col_dist(rgen);
    dist_mat.update(1, i, j);
    local_mirror(i, j) += 1;
  }

  dist_mat.commit();

  // perform consistency check
  cm::LocalMatrix<int> sum(2000, 1000);
  upcxx::reduce_one(local_mirror.data(), sum.data(), sum.m() * sum.n(),
                    upcxx::op_fast_add, 0)
      .wait();
  if (upcxx::rank_me() == 0) {
    for (long j = 0; j < sum.n(); ++j) {
      for (long i = 0; i < sum.m(); ++i) {
        ASSERT(sum(i, j), dist_mat(i, j), "Blah");
      }
    }
  }

  // note: ConvergentMatrix dtor contains an implicit barrier (none needed
  // here).
}

// TODO: Add similarly complex slice-based update.

}  // namespace test

int main(int argc, char *argv[]) {
  upcxx::init();
  test::randomElementUpdates();
  upcxx::finalize();
  return 0;
}
