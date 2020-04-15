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
  sum = 0;
  upcxx::reduce_all(local_mirror.data(), sum.data(), sum.m() * sum.n(),
                    upcxx::op_fast_add)
      .wait();
  auto success =
      dist_mat.verify_local_elements([&sum](int val, long i, long j) {
        if (sum(i, j) == val) return true;
        std::cerr << __FILE__ << ":" << __LINE__ << " @ " << upcxx::rank_me()
                  << "] verification failed at (" << i << ", " << j
                  << ") want: " << sum(i, j) << " got: " << val << std::endl;
        return false;
      });
  if (!success) {
    exit(1);
  }
}

void randomSliceUpdates() {
  constexpr int slice_m = 800, slice_n = 500;

  cm::LocalMatrix<int> local_mirror(2000, 1000);
  cm::ConvergentMatrix<int, NPROW, NPCOL, MB, NB,
                       /*LLD=*/1024>
      dist_mat(2000, 1000);

  std::random_device rdev;
  std::mt19937 rgen(rdev());
  std::uniform_int_distribution<long> row_dist(0, dist_mat.m() - 1);
  std::uniform_int_distribution<long> col_dist(0, dist_mat.n() - 1);

  std::vector<long> ix, jx;
  for (long i = 0; i < dist_mat.m(); ++i)
    ix.push_back(i);
  std::shuffle(ix.begin(), ix.end(), rgen);
  for (long j = 0; j < dist_mat.n(); ++j)
    jx.push_back(j);
  std::shuffle(jx.begin(), jx.end(), rgen);

  ix.resize(slice_m);
  jx.resize(slice_n);

  std::sort(ix.begin(), ix.end());
  std::sort(jx.begin(), jx.end());

  cm::LocalMatrix<int> slice(slice_m, slice_n);
  slice = 1;  // Set all elements to 1.

  for (int j = 0; j < slice_n; ++j)
    for (int i = 0; i < slice_m; ++i)
      local_mirror(ix[i], jx[j]) = 1;

  dist_mat.update(&slice, ix.data(), jx.data());
  dist_mat.commit();

  // perform consistency check
  cm::LocalMatrix<int> sum(2000, 1000);
  sum = 0;
  upcxx::reduce_all(local_mirror.data(), sum.data(), sum.m() * sum.n(),
                    upcxx::op_fast_add)
      .wait();
  auto success =
      dist_mat.verify_local_elements([&sum](int val, long i, long j) {
        if (sum(i, j) == val) return true;
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
  test::randomSliceUpdates();
  upcxx::finalize();
  return 0;
}
