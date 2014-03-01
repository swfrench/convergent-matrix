#include <cstdio>
#include <upcxx.h>

#include "convergent_matrix.hpp"

// global matrix dimensions
#define M 2000
#define N 1000

// block-cyclic distribution
#define NPCOL 2
#define NPROW 2
#define MB 64
#define NB 64
#define LLD 1024

// convenient typedef for the distributed matrix class
typedef cm::ConvergentMatrix<float, NPROW, NPCOL, MB, NB, LLD> cmat_t;

/**
 * A small example of some basic ConvergentMatrix functionality
 *
 * Must be run on 4 threads using the appropriate command (mpirun, aprun, etc.)
 * for the environment and GASNet conduit.
 */
void
simple()
{
  // initialize the distributed matrix abstraction
  cmat_t dist_mat( M, N );

  // optional: turn on replicated consistency checks (run during commit)
  // * requires enough memory to replicate the entire matrix to all threads
  // * requires a working MPI implementation
  dist_mat.consistency_check_on();

  // create and apply an update ...
  {
    // generate leading and trailing dimension indexing
    long ixs[] = { 0, 100, 1000, 1500 },
         jxs[] = { 0, 300, 600 };

    // generate matrix slice
    // indexing arrays above map into the global distributed matrix
    cm::LocalMatrix<float> A( 4, 3 );

    A = 1.0; // arbitrarily setting all elements to 1.0

    // apply the update (A may be destroyed when update() returns)
    dist_mat.update( &A, ixs, jxs );
  }

  // commit all updates
  dist_mat.commit();

  // get a raw pointer to local storage
  // * freed when dist_mat destroyed
  float * data_ptr = dist_mat.get_local_data();
  printf( "[%s] thread %3i : data_ptr[0] = %f\n",
          __func__, MYTHREAD, data_ptr[0] );

  // fetch the value at global index ( 0, 0 )
  // * should match data_ptr[0] on thread 0
  printf( "[%s] thread %3i : dist_mat(0, 0) = %f\n",
          __func__, MYTHREAD, dist_mat( 0, 0 ) );

  // sync before dist_mat goes out of scope and is destroyed
  upcxx::barrier();
}

int
main( int argc, char **argv )
{
  // initialize upcxx
  upcxx::init( &argc, &argv );

  // run the example above
  simple();

  // shut down upcxx
  upcxx::finalize();
}
