#include <cstdio>
#include <cstdlib>
#include <ctime>

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

// the test configuration
#include "simple_test_setup.hpp"

// convenient typedef for the distributed matrix class
typedef cm::ConvergentMatrix<real_t,NPROW,NPCOL,MB,NB,LLD> cmat_t;


/**
 * Initialize the test with random (symmetric) update indexing
 */
template <typename T, typename I>
int
setup( int rank, I **msub, I ***inds, T **data )
{
  int niter;
  I msub_max;

  // seed the rng
  srand( ( 1 + rank ) * time( NULL ) );

  // determine number of local iters
#if NITER_MIN == NITER_MAX
  niter = NITER_MIN;
#else
  niter = NITER_MIN + ( rand() % ( NITER_MAX - NITER_MIN ) );
#endif

  // allocate iter size / indexing
  *msub = new I[niter];
  *inds = new I * [niter];

  // derive sparse permuted updates
  msub_max = 0;
  for ( int i = 0; i < niter; i++ ) {
    // number of elems hit on this repetition
    int rep_size = MSUB_MIN + ( rand() % ( MSUB_MAX - MSUB_MIN ) );

    // total number of elems hit on this iter (MSUB_REPS repetitions)
    (*msub)[i] = MSUB_REPS * rep_size;

    // fill sparse index (repeating sub-structure)
    (*inds)[i] = new I[(*msub)[i]];
    for ( int r = 0; r < MSUB_REPS; r++ )
      if ( r == 0 )
        for ( int j = 0; j < rep_size; j++ )
          {
            I prop;
            bool ok = false;
            while ( ! ok )
              {
                ok = true;
                prop = rand() % MSUB_SIZE;
                for ( int k = 0; k < j; k++ )
                  if ( (*inds)[i][k] == prop )
                    {
                      ok = false;
                      break;
                    }
              }
            (*inds)[i][j] = prop;
          }
      else
        for ( int j = 0; j < rep_size; j++ )
          (*inds)[i][j + r * rep_size] =
            MSUB_SIZE + (*inds)[i][j + ( r - 1 ) * rep_size];

    // track largest update size
    msub_max = std::max( msub_max, (*msub)[i] );
  }

  printf( "[%s] max update dimension is %li\n", __func__, msub_max );

  // create fake data for update
  *data = new T[msub_max * msub_max];
  srand48( ( rank + 1 ) * time( NULL ) );
  for ( long i = 0; i < msub_max * msub_max; i++ )
    (*data)[i] = drand48();

  return niter;
}


/**
 * Run the test
 */
void
run()
{
  int niter;     // number of test-update rounds
  long *nxs;     // dimension of update (symmetric)
  long **ixs;    // update indexing
  real_t *data;  // update data (reused for each round, though indexing varies)

  // timing info for various stages
  double wt_commit = 0.0, wt_update = 0.0, wt_fill = 0.0;

  // initialize the distributed matrix
  cmat_t dist_mat( M, M );

  // generate test case
  niter = setup<real_t,long>( MYTHREAD, &nxs, &ixs, &data );

  printf( "Thread %4i | generated %i rounds of fake update indexing\n",
          MYTHREAD, niter );

  // set the progress interval to 1
  dist_mat.progress_interval( 1 );

  // turn on consistency checks (if available)
#if defined(ENABLE_CONSISTENCY_CHECK) && ! defined(FORCE_NO_CONSISTENCY_CHECK)
  dist_mat.consistency_check_on();
#endif

  printf( "Thread %4i | starting update rounds ...\n", MYTHREAD );

  // perform a number of dummy updates: symmetric case
  for ( int r = 0; r < niter; r++ ) {
    cm::LocalMatrix<real_t> GtG( nxs[r], nxs[r], data );
    // track update / binning time
    wt_update -= WTIME();
    dist_mat.update( &GtG, ixs[r] );
    wt_update += WTIME();
  }

  // commit all updates to the ConvergentMatrix abstraction
  printf( "Thread %4i | committing ...\n", MYTHREAD );

  // track commit time
  wt_commit -= WTIME();
  dist_mat.commit();
  wt_commit += WTIME();

  // commit all updates to the ConvergentMatrix abstraction
  printf( "Thread %4i | filling strictly lower triangular part ...\n",
          MYTHREAD );

  // track fill_lower / commit time
  wt_fill -= WTIME();
  dist_mat.fill_lower();
  dist_mat.commit();
  wt_fill += WTIME();

  // report timing
  printf( "Thread %4i | time spent in update: %fs\n",
          MYTHREAD, wt_update );
  printf( "Thread %4i | time spent in commit: %fs\n",
          MYTHREAD, wt_commit );
  printf( "Thread %4i | total time in initial update / commit phase: %fs\n",
          MYTHREAD, wt_update + wt_commit );
  printf( "Thread %4i | total time in fill / commit phase: %fs\n",
          MYTHREAD, wt_fill );

  // test the write functionality (if available)
#ifdef ENABLE_MPIIO_SUPPORT
  dist_mat.save( "test.matrix" );
#endif

  // fetch the local PBLAS-compatible block-cyclic storage array
  real_t * local_data = dist_mat.get_local_data();

  // display local leading entry
  printf( "Thread %4i | local_data[0] = %f\n",
          MYTHREAD, local_data[0] );

  // try remote random access read
  printf( "Thread %4i | dist_mat( 64, 64 ) = %f\n",
          MYTHREAD, dist_mat( 64, 64 ) );

  // synch before return and dist_mat is destroyed when it goes out of scope
  upcxx::barrier();
}

int
main( int argc, char **argv )
{
  // init upcxx
  upcxx::init( &argc, &argv );

  // init MPI
#ifdef MPI_INIT_REQUIRED
  int mpi_init;
  MPI_Initialized( &mpi_init );
  if ( ! mpi_init )
    MPI_Init( &argc, &argv );
#endif

  // run the test
  run();

#ifdef MPI_INIT_REQUIRED
  if ( ! mpi_init )
    MPI_Finalize();
#endif

  // shut down upcxx
  upcxx::finalize();

  return 0;
}
