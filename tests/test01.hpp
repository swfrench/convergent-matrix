#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <algorithm>

#define TESTNAME "test01"

template <typename T>
int
gen_test01( int rank, long **msub, long ***inds, T **data )
{
  int niter;
  long msub_max;

  // seed the rng
  srand( ( 1 + rank ) * time( NULL ) );

  // determine number of local iters
#if NITER_MIN == NITER_MAX
  niter = NITER_MIN;
#else
  niter = NITER_MIN + ( rand() % ( NITER_MAX - NITER_MIN ) );
#endif

  // allocate iter size / indexing
  *msub = new long[niter];
  *inds = new long * [niter];

  // derive sparse permuted updates
  msub_max = 0;
  for ( int i = 0; i < niter; i++ ) {
    // number of elems hit on this repetition
    int rep_size = MSUB_MIN + ( rand() % ( MSUB_MAX - MSUB_MIN ) );

    // total number of elems hit on this iter (MSUB_REPS repetitions)
    (*msub)[i] = MSUB_REPS * rep_size;

    // fill sparse index (repeating sub-structure)
    (*inds)[i] = new long[(*msub)[i]];
    for ( int r = 0; r < MSUB_REPS; r++ )
      if ( r == 0 )
        for ( int j = 0; j < rep_size; j++ )
          {
            long prop;
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
