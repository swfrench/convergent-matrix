#include <iostream>
#include <upcxx.h>
#include "convergent_matrix.hpp"

using namespace std;

int
main( int argc, char **argv )
{
  convergent::ConvergentMatrix<float> *dist_mat;
  convergent::LocalMatrix<float> *local_mat;

  // init upcxx
  upcxx::init( &argc, &argv );
 
  // init the eventually-consistent matrix abstraction
  dist_mat = new convergent::ConvergentMatrix<float>( 1000, 1000 );

  // perform a number of dummy updates from subsets of threads
  const int m = 100, n = 20; // matrix dims
  const int niter = 10;      // number of cycles
  for ( int iter = 0; iter < niter; iter++ )
    if ( MYTHREAD % 2 == iter % 2 )
      {
        // trailing index of G
        long *ix;
        // local dense matrices for G and G^tG
        convergent::LocalMatrix<float> *G, *GtG;

        // init dummy G matrix
        G = new convergent::LocalMatrix<float>( m, n, 1.0);

        // init dummy trailing index
        ix = new long [n];
        for ( int i = 0; i < n; i++ )
            ix[i] = 50 * i;

        // form G^t G
        GtG = (*G->trans()) * (*G);

        // schedule for (eventual) update via async remote tasks
        // ** only one index array supplied - assumed symmetric **
        dist_mat->update( GtG, ix );

        // clean up
        delete G;
        delete GtG;
        delete [] ix;
      }

  // freeze the ConvergentMatrix abstraction
  // ** implicit barrier() and wait() on remaining async updates **
  dist_mat->freeze();

  // fetch the local PBLAS-compatible block-cyclic storage array
  local_mat = dist_mat->as_local_matrix();

  // safe to delete dist_mat now
  delete dist_mat;

  // allow the LocalMatrix destructor to manage the underlying matrix data
  local_mat->override_free();

  // ** now we can now freely use local_mat->data() w/ the PBLAS and friends **

  // expected result on thread 0 at (0,0) for the above test
  if ( MYTHREAD == 0 )
    assert( (int)(*local_mat)( 0, 0 ) == ( THREADS / 2 ) * niter * m );

  cout << "Thread " << MYTHREAD << " "
       << "local_mat(0,0) = " << (*local_mat)( 0, 0 )
       << endl;

  // shut down upcxx
  upcxx::finalize();

  return 0;
}
