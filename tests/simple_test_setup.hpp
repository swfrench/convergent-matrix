#pragma once

// floating point type
typedef double real_t;

// number of update iterations
#define NITER_MIN 10
#define NITER_MAX 10

// M = 10000; 2 x 2 process grid
#define MSUB_REPS 10
#define MSUB_SIZE 1000
#define M ( MSUB_REPS * MSUB_SIZE )
#define MSUB_MIN 100
#define MSUB_MAX 120
#define NPCOL 2
#define NPROW 2
#define MB 64
#define NB 64
#define LLD 5056

// M = 100000; 8 x 8 process grid
//#define MSUB_REPS 10
//#define MSUB_SIZE 10000
//#define M ( MSUB_REPS * MSUB_SIZE )
//#define MSUB_MIN 1000
//#define MSUB_MAX 1200
//#define NPCOL 8
//#define NPROW 8
//#define MB 64
//#define NB 64
//#define LLD 12544
//#define FORCE_NO_CONSISTENCY_CHECK
