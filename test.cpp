#include <iostream>
#include <upcxx.h>

#include "convergent_matrix.hpp"

using namespace std;

int
main( int argc, char **argv )
{
  convergent::ConvergentMatrix<float> *mat;
  
  // init upcxx
  upcxx::init( &argc, &argv );
 
  // init the eventually-consistent matrix abstraction
  mat = new convergent::ConvergentMatrix<float>( 1000, 1000 );

  // perform a dummy update from a single thread
  if ( MYTHREAD == 1 )
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

  // finalize the ConvergentMatrix abstraction
  // ** implicit barrier() and wait() on async updates **
  mat->finalize();

  // shut down upcxx
  upcxx::finalize();

  return 0;
}
