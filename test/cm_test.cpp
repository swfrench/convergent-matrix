#include <cstdlib>
#include <iostream>
#include <random>

#include <upcxx/upcxx.hpp>

#include "include/convergent_matrix.hpp"
#include "include/local_matrix.hpp"

#define MB 64
#define NB 64
#define NPROW 2
#define NPCOL 2

namespace test {

// TODO: Add coverage of:
// - multi-epoch updates (i.e. multiple calls to commit)
// - slice-based updates (rather than single-element)
// - symmetric case (i.e. bulk lower triangular fill)

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
  upcxx::reduce_all(local_mirror.data(), sum.data(), sum.m() * sum.n(),
                    upcxx::op_fast_add)
      .wait();
  auto success =
      dist_mat.verify_local_elements([&sum](int val, long i, long j) {
        if (sum(i, j) != val) return true;
        std::cerr << __FILE__ << ":" << __LINE__ << " @ " << upcxx::rank_me()
                  << "] verification failed at (" << i << ", " << j
                  << ") want: " << sum(i, j) << " got: " << val << std::endl;
        return false;
      });
  if (!success) {
    exit(1);
  }
}

}  // namespace test

int main(int argc, char *argv[]) {
  upcxx::init();
  test::randomElementUpdates();
  upcxx::finalize();
  return 0;
}
