#include <iostream>
#include <upcxx.h>
#include "convergent_matrix.hpp"

using namespace std;

int
main( int argc, char **argv )
{
  convergent::ConvergentMatrix<float> *mat;
  convergent::LocalMatrix<float> *local_mat;

  // init upcxx
  upcxx::init( &argc, &argv );
 
  // init the eventually-consistent matrix abstraction
  mat = new convergent::ConvergentMatrix<float>( 1000, 1000 );

  // perform a dummy updates from a subset of threads
  if ( MYTHREAD == 1 || MYTHREAD == 3 )
    {
      // dims
      const int m = 100, n = 20;
      // trailing index of G
      long *ix;
      // local dense matrices for G and G^tG
      convergent::LocalMatrix<float> *G, *GtG;

      // init dummy G matrix
      G = new convergent::LocalMatrix<float>( m, n, 1.0);

      // init dummy trailing index
      ix = new long [n];
      for ( int i = 0; i < n; i++ )
        ix[i] = 10 * i;

      // form G^t G
      GtG = (*G->trans()) * (*G);

      // schedule for (eventual) update via async remote tasks
      // ** only one index array supplied - assumed symmetric **
      mat->update( GtG, ix );
    }

  // freeze the ConvergentMatrix abstraction
  // ** implicit barrier() and wait() on remaining async updates **
  mat->freeze();

  // fetch the local PBLAS-compatible block-cyclic storage array
  local_mat = mat->as_local_matrix();

  // safe to delete mat now
  delete mat;

  // allow the LocalMatrix destructor to manage the underlying matrix data
  local_mat->override_free();

  // ** now we can now freely use local_mat->data() w/ the PBLAS and friends **

  cout << "Thread " << MYTHREAD << " "
       << "local_mat(0,0) = " << (*local_mat)( 0, 0 )
       << endl;

  // shut down upcxx
  upcxx::finalize();

  return 0;
}
