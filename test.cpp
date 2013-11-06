#include <iostream>
#include <upcxx.h>

#include "convergent_matrix.hpp"

using namespace std;

int
main( int argc, char **argv )
{
  convergent::ConvergentMatrix<float> *mat;
  upcxx::init( &argc, &argv );
  mat = new convergent::ConvergentMatrix<float>( 1000, 1000 );
  if ( MYTHREAD == 1 )
    {
      const int m = 20;
      long *ix;
      convergent::LocalMatrix<float> *Mat;
      Mat = new convergent::LocalMatrix<float>( m, m, 1.0);
      ix = new long [m];
      for ( int i = 0; i < m; i++ )
        ix[i] = 10 * i;
      mat->update( Mat, ix );
    }
  mat->finalize();
  upcxx::finalize();
  return 0;
}
