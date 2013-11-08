#include <iostream>
#include <upcxx.h>
#include <cstdlib>
#include <ctime>

#include "convergent_matrix.hpp"

// global matrix size
#define M 100000

// block-cyclic distribution
#define NPCOL 24
#define NPROW 24
#define MB 64
#define NB 64
#define LLD 4500
typedef cm::ConvergentMatrix<float,NPROW,NPCOL,MB,NB,LLD> cmat_t;

// number of artificial update rounds
#define NUM_ROUNDS 10

// dimension of update
#define MSUB_MIN 10000
#define MSUB_MAX 15000

using namespace std;

int
main( int argc, char **argv )
{
  cmat_t *dist_mat;
  cm::LocalMatrix<float> *local_mat;

  // init upcxx
  upcxx::init( &argc, &argv );

  // generate fake update rounds
  srand( time(NULL) );
  long *nxs = new long[NUM_ROUNDS];
  long **ixs = new long *[NUM_ROUNDS];
  for ( int r = 0; r < NUM_ROUNDS; r++ )
    {
      nxs[r] = MSUB_MIN + rand() % (MSUB_MAX - MSUB_MIN);
      ixs[r] = new long[nxs[r]];
      ixs[r][0] = rand() % 5;
      for ( int i = 1; i < nxs[r]; i++ )
        ixs[r][i] = ixs[r][i-1] + rand() % 5;
      cout << "Round " << r << " "
           << "size: " << nxs[r]
           << endl;
    }
  
 
  // init distributed matrix object (block-cyclic: see convergent_matrix.hpp)
  dist_mat = new cmat_t( M, M );

  // perform a number of dummy updates
  for ( int r = 0; r < NUM_ROUNDS; r++ )
    {
      cm::LocalMatrix<float> *GtG;

      GtG = new cm::LocalMatrix<float>( nxs[r], nxs[r], 1.0 );

      dist_mat->update( GtG, ixs[r] );

      usleep( rand() % 500000 ); // sleep up to 500 ms for "computation"

      delete GtG;
    }

  // freeze the ConvergentMatrix abstraction
  dist_mat->freeze();

  // fetch the local PBLAS-compatible block-cyclic storage array
  local_mat = dist_mat->as_local_matrix();
  local_mat->override_free();

  // ** we can now freely use local_mat->data() w/ the PBLAS and friends **

  // safe to delete dist_mat now
  delete dist_mat;

  // shut down upcxx
  upcxx::finalize();

  return 0;
}
