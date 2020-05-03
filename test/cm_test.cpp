// Tests for basic functionality of ConvergentMatrix.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#ifdef ENABLE_MPIIO_SUPPORT
#include <mpi.h>
#endif

#include <upcxx/upcxx.hpp>

#include "include/convergent_matrix.hpp"
#include "include/local_matrix.hpp"

#define MB 64
#define NB 64
#define NPROW 2
#define NPCOL 2

#define LOG \
  std::cerr << __FILE__ << ":" << __LINE__ << " @ " << upcxx::rank_me() << "] "

namespace test {

namespace {

bool match(int a, int b) { return a == b; }

bool match(float a, float b) {
  constexpr float tol = 1.0e-7;
  const float delta = std::abs(a - b), ref = std::max(std::abs(a), std::abs(b));
  if (ref > 0.0) {
    return delta / ref < tol;
  }
  return true;  // both are zero.
}

}  // namespace

template <typename T>
void randomElementUpdates() {
  constexpr int niter = 1000, nupdate = 10000;

  cm::LocalMatrix<T> local_mirror(2000, 1000);
  cm::ConvergentMatrix<T, NPROW, NPCOL, MB, NB,
                       /*LLD=*/1024>
      dist_mat(2000, 1000);

  std::random_device rdev;
  std::mt19937 rgen(rdev());
  std::uniform_int_distribution<long> row_dist(0, dist_mat.m() - 1);
  std::uniform_int_distribution<long> col_dist(0, dist_mat.n() - 1);

  local_mirror = 0;
  for (int iter = 0; iter < niter; ++iter) {
    for (int n = 0; n < nupdate; ++n) {
      const long i = row_dist(rgen), j = col_dist(rgen);
      dist_mat.update(1, i, j);
      local_mirror(i, j) += 1;
    }

    dist_mat.commit();

    cm::LocalMatrix<T> sum(2000, 1000);
    sum = 0;
    upcxx::reduce_all(local_mirror.data(), sum.data(), sum.m() * sum.n(),
                      upcxx::op_fast_add)
        .wait();
    auto success =
        dist_mat.verify_local_elements([&sum, iter](T val, long i, long j) {
          if (match(sum(i, j), val)) return true;
          LOG << "verification failed at (" << i << ", " << j
              << ") for update epoch " << iter << " want: " << sum(i, j)
              << " got: " << val << std::endl;
          return false;
        });
    if (!success) {
      LOG << __func__ << " failed. Exiting ..." << std::endl;
      exit(1);
    }

    // Wait for verification to complete across all processes before starting
    // the next round (lest we potentially observe remotely injected updates).
    upcxx::barrier();
  }
}

template <typename T>
void randomElementUpdatesSymmetricFill() {
  constexpr int niter = 1000, nupdate = 10000;

  cm::LocalMatrix<T> local_mirror(2000, 2000);
  cm::ConvergentMatrix<T, NPROW, NPCOL, MB, NB,
                       /*LLD=*/1024>
      dist_mat(2000, 2000);

  std::random_device rdev;
  std::mt19937 rgen(rdev());
  std::uniform_int_distribution<long> row_dist(0, dist_mat.m() - 1);
  std::uniform_int_distribution<long> col_dist(0, dist_mat.n() - 1);

  local_mirror = 0;
  for (int iter = 0; iter < niter; ++iter) {
    for (int n = 0; n < nupdate;) {
      const long i = row_dist(rgen), j = col_dist(rgen);
      // Fill only the upper trianguler part of dist_mat ...
      if (j >= i) {
        dist_mat.update(1, i, j);
        // ... but maintain the full symmetric matrix in local_mirror.
        local_mirror(i, j) += 1;
        if (i != j) local_mirror(j, i) += 1;
        ++n;
      }
    }

    dist_mat.commit();
  }

  // Note: We do not perform verification following each update epoch (since
  // repeated application of fill_lower does not make sense, given that its not
  // idempotent).

  dist_mat.fill_lower();

  // fill_lower will trigger remote updates, and thus requires a commit call.
  dist_mat.commit();

  cm::LocalMatrix<T> sum(2000, 2000);
  sum = 0;
  upcxx::reduce_all(local_mirror.data(), sum.data(), sum.m() * sum.n(),
                    upcxx::op_fast_add)
      .wait();
  auto success = dist_mat.verify_local_elements([&sum](T val, long i, long j) {
    if (match(sum(i, j), val)) return true;
    LOG << "verification failed at (" << i << ", " << j
        << ") want: " << sum(i, j) << " got: " << val << std::endl;
    return false;
  });
  if (!success) {
    LOG << __func__ << " failed. Exiting ..." << std::endl;
    exit(1);
  }
}

template <typename T>
void randomSliceUpdates() {
  constexpr int niter = 1000, slice_m = 800, slice_n = 500;

  cm::LocalMatrix<T> local_mirror(2000, 1000);
  cm::ConvergentMatrix<T, NPROW, NPCOL, MB, NB,
                       /*LLD=*/1024>
      dist_mat(2000, 1000);

  std::random_device rdev;
  std::mt19937 rgen(rdev());

  cm::LocalMatrix<T> slice(slice_m, slice_n);
  slice = 1;  // Set all elements to 1.

  local_mirror = 0;
  for (int iter = 0; iter < niter; ++iter) {
    std::vector<long> ix, jx;
    for (long i = 0; i < dist_mat.m(); ++i) ix.push_back(i);
    for (long j = 0; j < dist_mat.n(); ++j) jx.push_back(j);

    std::shuffle(ix.begin(), ix.end(), rgen);
    std::shuffle(jx.begin(), jx.end(), rgen);

    ix.resize(slice_m);
    jx.resize(slice_n);

    std::sort(ix.begin(), ix.end());
    std::sort(jx.begin(), jx.end());

    for (int j = 0; j < slice_n; ++j)
      for (int i = 0; i < slice_m; ++i) local_mirror(ix[i], jx[j]) += 1;

    dist_mat.update(&slice, ix.data(), jx.data());
    dist_mat.commit();

    cm::LocalMatrix<T> sum(2000, 1000);
    sum = 0;
    upcxx::reduce_all(local_mirror.data(), sum.data(), sum.m() * sum.n(),
                      upcxx::op_fast_add)
        .wait();
    auto success =
        dist_mat.verify_local_elements([&sum, iter](T val, long i, long j) {
          if (match(sum(i, j), val)) return true;
          LOG << "verification failed at (" << i << ", " << j
              << ") for update epoch " << iter << " want: " << sum(i, j)
              << " got: " << val << std::endl;
          return false;
        });
    if (!success) {
      LOG << __func__ << " failed. Exiting ..." << std::endl;
      exit(1);
    }

    // Wait for verification to complete across all processes before starting
    // the next round (lest we potentially observe remotely injected updates).
    upcxx::barrier();
  }
}

template <typename T>
void randomSliceUpdatesSymmetricFill() {
  constexpr int niter = 1000, slice_n = 500;

  cm::LocalMatrix<T> local_mirror(2000, 2000);
  cm::ConvergentMatrix<T, NPROW, NPCOL, MB, NB,
                       /*LLD=*/1024>
      dist_mat(2000, 2000);

  std::random_device rdev;
  std::mt19937 rgen(rdev());

  cm::LocalMatrix<T> slice(slice_n, slice_n);
  slice = 0;  // Upper trianguler slice.
  for (int j = 0; j < slice_n; ++j)
    for (int i = 0; i <= j; ++i) slice(i, j) = 1;

  local_mirror = 0;
  for (int iter = 0; iter < niter; ++iter) {
    std::vector<long> ix;
    for (long i = 0; i < dist_mat.m(); ++i) ix.push_back(i);

    std::shuffle(ix.begin(), ix.end(), rgen);

    ix.resize(slice_n);

    std::sort(ix.begin(), ix.end());

    for (int j = 0; j < slice_n; ++j)
      for (int i = 0; i <= j; ++i) {
        local_mirror(ix[i], ix[j]) += 1;
        if (i != j) local_mirror(ix[j], ix[i]) += 1;
      }

    dist_mat.update(&slice, ix.data());
    dist_mat.commit();
  }

  // Note: Unlike the other tests, we do not perform verification following
  // each update epoch (since repeated application of fill_lower does not make
  // sense, given that its not idempotent).

  dist_mat.fill_lower();

  // fill_lower will trigger remote updates, and thus requires a commit call.
  dist_mat.commit();

  cm::LocalMatrix<T> sum(2000, 2000);
  sum = 0;
  upcxx::reduce_all(local_mirror.data(), sum.data(), sum.m() * sum.n(),
                    upcxx::op_fast_add)
      .wait();
  auto success = dist_mat.verify_local_elements([&sum](T val, long i, long j) {
    if (match(sum(i, j), val)) return true;
    LOG << "verification failed at (" << i << ", " << j
        << ") want: " << sum(i, j) << " got: " << val << std::endl;
    return false;
  });
  if (!success) {
    LOG << __func__ << " failed. Exiting ..." << std::endl;
    exit(1);
  }
}

#ifdef ENABLE_MPIIO_SUPPORT

template <typename T>
void randomElementUpdatesMPIIO() {
  constexpr int nupdate = 10000;

  cm::LocalMatrix<T> local_mirror(2000, 1000);
  cm::ConvergentMatrix<T, NPROW, NPCOL, MB, NB,
                       /*LLD=*/1024>
      dist_mat1(2000, 1000);

  std::random_device rdev;
  std::mt19937 rgen(rdev());
  std::uniform_int_distribution<long> row_dist(0, dist_mat1.m() - 1);
  std::uniform_int_distribution<long> col_dist(0, dist_mat1.n() - 1);

  local_mirror = 0;
  for (int n = 0; n < nupdate; ++n) {
    const long i = row_dist(rgen), j = col_dist(rgen);
    dist_mat1.update(1, i, j);
    local_mirror(i, j) += 1;
  }

  dist_mat1.commit();

  cm::LocalMatrix<T> sum(2000, 1000);
  sum = 0;
  upcxx::reduce_all(local_mirror.data(), sum.data(), sum.m() * sum.n(),
                    upcxx::op_fast_add)
      .wait();
  auto success = dist_mat1.verify_local_elements([&sum](T val, long i, long j) {
    if (match(sum(i, j), val)) return true;
    LOG << "verification failed at (" << i << ", " << j
        << ") during initial construction; want: " << sum(i, j)
        << " got: " << val << std::endl;
    return false;
  });
  if (!success) {
    LOG << __func__ << " failed. exiting ..." << std::endl;
    exit(1);
  }

  // write out the current state ...
  const char* save_dir = std::getenv("CM_TEST_DIR");
  std::string save_path = save_dir == nullptr ? "/tmp" : save_dir;
  save_path += "/dist_mat1";
  dist_mat1.save(save_path.c_str());

  cm::ConvergentMatrix<T, NPROW, NPCOL, MB, NB, 1024> dist_mat2(2000, 1000);

  // ... and read it back in again ...
  dist_mat2.load(save_path.c_str());

  // ... and re-verify.
  success = dist_mat2.verify_local_elements([&sum](T val, long i, long j) {
    if (match(sum(i, j), val)) return true;
    LOG << "verification failed at (" << i << ", " << j
        << ") after load from disk; want: " << sum(i, j) << " got: " << val
        << std::endl;
    return false;
  });
  if (!success) {
    LOG << __func__ << " failed. Exiting ..." << std::endl;
    exit(1);
  }
}

#endif  // ENABLE_MPIIO_SUPPORT

template <typename T>
void runAllTests() {
  test::randomElementUpdates<T>();
  test::randomElementUpdatesSymmetricFill<T>();
  test::randomSliceUpdates<T>();
  test::randomSliceUpdatesSymmetricFill<T>();
#ifdef ENABLE_MPIIO_SUPPORT
  test::randomElementUpdatesMPIIO<T>();
#endif
}

}  // namespace test

int main(int argc, char* argv[]) {
  upcxx::init();

#ifdef ENABLE_MPIIO_SUPPORT
  int init = 0;
  if (MPI_Initialized(&init) != MPI_SUCCESS) {
    LOG << "could not determine mpi initialization status." << std::endl;
    exit(1);
  } else if (!init && MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    LOG << "could not initialize mpi." << std::endl;
    exit(1);
  }
#endif

  test::runAllTests<int>();
  test::runAllTests<float>();

#ifdef ENABLE_MPIIO_SUPPORT
  if (!init) {
    MPI_Finalize();
  }
#endif

  upcxx::finalize();
  return 0;
}
