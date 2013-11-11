#include <iostream>
#include <upcxx.h>
#include <cstdlib>
#include <ctime>

#include "convergent_matrix.hpp"

// global matrix size
#define M 100000

// block-cyclic distribution
#define NPCOL 4
#define NPROW 4
#define MB 64
#define NB 64
#define LLD 25024
typedef cm::ConvergentMatrix<float,NPROW,NPCOL,MB,NB,LLD> cmat_t;

// number of artificial update rounds
#define NUM_ROUNDS 100

// dimension of update
#define MSUB_MIN 10000
#define MSUB_MAX 15000
#define MSUB_MAX_STEP 5

using namespace std;

int
main( int argc, char **argv )
{
  cmat_t *dist_mat;
  cm::LocalMatrix<float> *local_mat;

  // init upcxx
  upcxx::init( &argc, &argv );

  if ( MYTHREAD == 0 )
    cout << "NUM_ROUNDS : " << NUM_ROUNDS << endl;

  // generate fake update rounds
  srand( MYTHREAD * time(NULL) );
  long *nxs = new long[NUM_ROUNDS];
  long **ixs = new long *[NUM_ROUNDS];
  for ( int r = 0; r < NUM_ROUNDS; r++ )
    {
      nxs[r] = MSUB_MIN + rand() % (MSUB_MAX - MSUB_MIN);
      ixs[r] = new long[nxs[r]];
      ixs[r][0] = 0; // always hit i = 0 (for testing)
      for ( int i = 1; i < nxs[r]; i++ )
        ixs[r][i] = ixs[r][i-1] + (1 + rand() % MSUB_MAX_STEP);
    }
  
 
  // init distributed matrix object (block-cyclic: see convergent_matrix.hpp)
  dist_mat = new cmat_t( M, M );
  dist_mat->progress_interval( 1 );

  // perform a number of dummy updates
  for ( int r = 0; r < NUM_ROUNDS; r++ )
    {
      cm::LocalMatrix<float> *GtG;

      GtG = new cm::LocalMatrix<float>( nxs[r], nxs[r], 1.0 );

      dist_mat->update( GtG, ixs[r] );
      delete GtG;
      delete [] ixs[r];

      usleep( rand() % 500000 ); // sleep up to 500 ms for "computation"
    }

  // freeze the ConvergentMatrix abstraction
  cout << "Thread " << MYTHREAD << " freezing ... " << endl;
  dist_mat->freeze();

  // fetch the local PBLAS-compatible block-cyclic storage array
  local_mat = dist_mat->as_local_matrix();
  local_mat->override_free();
  if ( MYTHREAD == 0 )
    cout << "TEST : local_mat(0,0) = " << (*local_mat)(0,0) << endl;
  upcxx::barrier();

  // ** we can now freely use local_mat->data() w/ the PBLAS and friends **

  // safe to delete dist_mat now
  delete dist_mat;

  // shut down upcxx
  upcxx::finalize();

  return 0;
}
