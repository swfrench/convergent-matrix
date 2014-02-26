#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cassert>

#if ( defined(USE_MPI_WTIME) || \
      defined(ENABLE_CONSISTENCY_CHECK) || \
      defined(ENABLE_MPIIO_SUPPORT) )
#include <mpi.h>
#define MPI_INIT_REQUIRED
#endif

#ifndef USE_MPI_WTIME
#include <omp.h>
#define WTIME omp_get_wtime
#else
#define WTIME MPI_Wtime
#endif

#include <upcxx.h>

#include "convergent_matrix.hpp"

/**
 * Define test config
 */

#define NITER_MIN 10
#define NITER_MAX 10

// 100000 for 8 x 8
//#define MSUB_REPS 10
//#define MSUB_SIZE 10000
//#define M ( MSUB_REPS * MSUB_SIZE )
//#define MSUB_MIN 1000
//#define MSUB_MAX 1200

// 25000 for 2 x 2
//#define MSUB_REPS 10
//#define MSUB_SIZE 2500
//#define M ( MSUB_REPS * MSUB_SIZE )
//#define MSUB_MIN 250
//#define MSUB_MAX 300

// 10000 for 2 x 2
#define MSUB_REPS 10
#define MSUB_SIZE 1000
#define M ( MSUB_REPS * MSUB_SIZE )
#define MSUB_MIN 100
#define MSUB_MAX 120

#include "tests/test01.hpp"

/**
 * Done w/ test config
 */

// block-cyclic distribution
#define NPCOL 2
#define NPROW 2
#define MB 64
#define NB 64
#define LLD 12544
typedef cm::ConvergentMatrix<double,NPROW,NPCOL,MB,NB,LLD> cmat_t;

using namespace std;

int
main( int argc, char **argv )
{
  cmat_t *dist_mat;
  int niter;
  long *nxs, **ixs;
  double *data;
  double *local_data;

  // init upcxx
  upcxx::init( &argc, &argv );

  assert( THREADS == NPROW * NPCOL );

  // init MPI for tests
#ifdef MPI_INIT_REQUIRED
  int mpi_init;
  MPI_Initialized( &mpi_init );
  if ( ! mpi_init )
    MPI_Init( &argc, &argv );
#endif

  printf( "%4i : up - using test \"%s\"\n", MYTHREAD, TESTNAME ); fflush( stdout );

  // generate test case
  niter = gen_test01<double, long>( MYTHREAD, &nxs, &ixs, &data );

  printf( "%4i : generated %i rounds of fake update indexing\n",
          MYTHREAD, niter ); fflush( stdout );

  // init distributed matrix object (block-cyclic: see convergent_matrix.hpp)
  dist_mat = new cmat_t( M, M );
  dist_mat->progress_interval( 1 );
#ifdef ENABLE_CONSISTENCY_CHECK
  dist_mat->consistency_check_on();
#endif
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
      wt_tot -= WTIME();
      dist_mat->update( GtG, ixs[r] );
      wt_tot += WTIME();
      delete GtG;
    }

  // commit all updates to the ConvergentMatrix abstraction
  printf( "%4i : committing ...\n", MYTHREAD ); fflush( stdout );
  // track commit time
  wt_tot -= WTIME();
  dist_mat->commit();
  wt_tot += WTIME();
  printf( "%4i : total time spent in update / commit %fs\n", MYTHREAD, wt_tot ); fflush( stdout );

  // test the write functionality
#ifdef ENABLE_MPIIO_SUPPORT
  dist_mat->save( "test.matrix" );
#endif

  // fetch the local PBLAS-compatible block-cyclic storage array
  local_data = dist_mat->get_local_data();
  printf( "%4i : local_data[0] = %f\n", MYTHREAD, local_data[0] );
  printf( "%4i : dist_mat( 64, 64 ) = %f\n", MYTHREAD, (*dist_mat)( 64, 64 ) );
  upcxx::barrier();

  // safe to delete dist_mat now
  delete dist_mat;

#ifdef MPI_INIT_REQUIRED
  MPI_Finalize();
#endif

  // shut down upcxx
  upcxx::finalize();

  return 0;
}
