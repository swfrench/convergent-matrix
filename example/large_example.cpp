#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cassert>
#include <upcxx.h>

#include "convergent_matrix.hpp"

// global matrix size
#define M 100000

// block-cyclic distribution
#define NPCOL 8
#define NPROW 8
#define MB 64
#define NB 64
#define LLD 12544
typedef cm::ConvergentMatrix<double,NPROW,NPCOL,MB,NB,LLD> cmat_t;

// number of artificial update rounds
#define NUM_ROUNDS 10

// dimension of update
#define MSUB_MIN 10000
#define MSUB_MAX 15000
#define MSUB_MAX_STEP 5

using namespace std;

int
main( int argc, char **argv )
{
  cmat_t *dist_mat;
  cm::LocalMatrix<double> *local_mat;

  // init upcxx
  upcxx::init( &argc, &argv );

  assert( THREADS == NPROW * NPCOL );

  printf( "%4i : up\n", MYTHREAD ); fflush( stdout );

  // generate fake update rounds
  srand( MYTHREAD + time(NULL) );
  long nxs_max = 0;
  long *nxs = new long[NUM_ROUNDS];
  long **ixs = new long *[NUM_ROUNDS];
  for ( int r = 0; r < NUM_ROUNDS; r++ )
    {
      nxs[r] = MSUB_MIN + rand() % (MSUB_MAX - MSUB_MIN);
      ixs[r] = new long[nxs[r]];
      ixs[r][0] = 0; // always hit i = 0 (for testing)
      for ( long i = 1; i < nxs[r]; i++ )
        ixs[r][i] = ixs[r][i-1] + (1 + rand() % MSUB_MAX_STEP);
      nxs_max = max( nxs_max, nxs[r] );
    }

  printf( "%4i : generated %i rounds of fake update indexing (max local size: %li)\n",
          MYTHREAD, NUM_ROUNDS, nxs_max ); fflush( stdout );

  // now allocate fake data
  srand48( MYTHREAD + time(NULL) );
  double *data = new double[nxs_max * nxs_max];
  double data_max = 0.0;
  for ( long ij = 0; ij < nxs_max * nxs_max; ij++ )
    {
      data[ij] = drand48();
      data_max = max( data[ij], data_max );
    }

  printf( "%4i : populated update w/ random data (max: %f)\n",
          MYTHREAD, data_max ); fflush( stdout );

  // init distributed matrix object (block-cyclic: see convergent_matrix.hpp)
  dist_mat = new cmat_t( M, M );
  dist_mat->progress_interval( 1 );
  printf( "%4i : initialized distributed matrix (progress interval: %i)\n",
          MYTHREAD, dist_mat->progress_interval() ); fflush( stdout );

  printf( "%4i : starting update rounds ...\n", MYTHREAD ); fflush( stdout );

  // perform a number of dummy updates
  for ( int r = 0; r < NUM_ROUNDS; r++ )
    {
      cm::LocalMatrix<double> *GtG;
      GtG = new cm::LocalMatrix<double>( nxs[r], nxs[r], data );
      dist_mat->update( GtG, ixs[r] );
      delete GtG;
    }

  // freeze the ConvergentMatrix abstraction
  printf( "%4i : freezing ...\n", MYTHREAD ); fflush( stdout );
  dist_mat->freeze();

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
