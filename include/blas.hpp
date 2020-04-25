#pragma once

/**
 * Overloaded gemm() and gemv() operations for float and double
 */

extern "C" void sgemv_(char *t, int *m, int *n, float *alpha, float *a,
                       int *lda, float *x, int *incx, float *beta, float *y,
                       int *incy);

extern "C" void dgemv_(char *t, int *m, int *n, double *alpha, double *a,
                       int *lda, double *x, int *incx, double *beta, double *y,
                       int *incy);

inline void gemv(char *t, int *m, int *n, float *alpha, float *a, int *lda,
                 float *x, int *incx, float *beta, float *y, int *incy) {
  sgemv_(t, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

inline void gemv(char *t, int *m, int *n, double *alpha, double *a, int *lda,
                 double *x, int *incx, double *beta, double *y, int *incy) {
  dgemv_(t, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

extern "C" void sgemm_(char *ta, char *tb, int *m, int *n, int *k, float *alpha,
                       float *a, int *lda, float *b, int *ldb, float *beta,
                       float *c, int *ldc);

extern "C" void dgemm_(char *ta, char *tb, int *m, int *n, int *k,
                       double *alpha, double *a, int *lda, double *b, int *ldb,
                       double *beta, double *c, int *ldc);

inline void gemm(char *ta, char *tb, int *m, int *n, int *k, double *alpha,
                 double *a, int *lda, double *b, int *ldb, double *beta,
                 double *c, int *ldc) {
  dgemm_(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

inline void gemm(char *ta, char *tb, int *m, int *n, int *k, float *alpha,
                 float *a, int *lda, float *b, int *ldb, float *beta, float *c,
                 int *ldc) {
  sgemm_(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
