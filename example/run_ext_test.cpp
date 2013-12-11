#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cassert>
#include <omp.h>
#include <upcxx.h>

#include "convergent_matrix.hpp"

/**
 * Define test config
 */

#define NITER_MIN 10
#define NITER_MAX 10

// 100000 for 8 x 8
#define MSUB_REPS 10
#define MSUB_SIZE 10000
#define M ( MSUB_REPS * MSUB_SIZE )
#define MSUB_MIN 1000
#define MSUB_MAX 1200

// 25000 for 2 x 2
//#define MSUB_REPS 10
//#define MSUB_SIZE 2500
//#define M ( MSUB_REPS * MSUB_SIZE )
//#define MSUB_MIN 250
//#define MSUB_MAX 300

#include "tests/test01.hpp"

/**
 * Done w/ test config
 */

// block-cyclic distribution
#define NPCOL 8
#define NPROW 8
#define MB 64
#define NB 64
#define LLD 12544
typedef cm::ConvergentMatrix<double,NPROW,NPCOL,MB,NB,LLD> cmat_t;

using namespace std;

int
main( int argc, char **argv )
{
  cmat_t *dist_mat;
  cm::LocalMatrix<double> *local_mat;
  int niter;
  long *nxs, **ixs;
  double *data;

  // init upcxx
  upcxx::init( &argc, &argv );

  assert( THREADS == NPROW * NPCOL );

  printf( "%4i : up - using test \"%s\"\n", MYTHREAD, TESTNAME ); fflush( stdout );

  // generate test case
  niter = gen_test01<double>( MYTHREAD, &nxs, &ixs, &data );

  printf( "%4i : generated %i rounds of fake update indexing\n",
          MYTHREAD, niter ); fflush( stdout );

  // init distributed matrix object (block-cyclic: see convergent_matrix.hpp)
  dist_mat = new cmat_t( M, M );
  dist_mat->progress_interval( 1 );
  printf( "%4i : initialized distributed matrix (progress interval: %i)\n",
          MYTHREAD, dist_mat->progress_interval() ); fflush( stdout );

  printf( "%4i : starting update rounds ...\n", MYTHREAD ); fflush( stdout );

  double wt_tot = 0.0;

  // perform a number of dummy updates
  for ( int r = 0; r < niter; r++ )
    {
      cm::LocalMatrix<double> *GtG;
      GtG = new cm::LocalMatrix<double>( nxs[r], nxs[r], data );
      // track update time
      wt_tot -= omp_get_wtime();
      dist_mat->update( GtG, ixs[r] );
      wt_tot += omp_get_wtime();
      delete GtG;
    }

  // freeze the ConvergentMatrix abstraction
  printf( "%4i : freezing ...\n", MYTHREAD ); fflush( stdout );
  // track freeze time
  wt_tot -= omp_get_wtime();
  dist_mat->freeze();
  wt_tot += omp_get_wtime();
  printf( "%4i : total time spent in update / freeze %fs\n", MYTHREAD, wt_tot ); fflush( stdout );

  // fetch the local PBLAS-compatible block-cyclic storage array
  local_mat = dist_mat->as_local_matrix();
  local_mat->override_free();
  printf( "%4i : local_mat(0,0) = %f\n", MYTHREAD, (*local_mat)(0,0) );
  upcxx::barrier();

  // safe to delete dist_mat now
  delete dist_mat;

  // shut down upcxx
  upcxx::finalize();

  return 0;
}
