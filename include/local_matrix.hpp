#pragma once

// overloaded definitions of gemm() and gemv() for double and float
#include "blas.hpp"

namespace cm {

/**
 * A local dense matrix abstraction supporting a limited set of level 2 and 3
 * BLAS operations and element-wise arithmetic
 */
template <typename T>
class LocalMatrix {
 private:
  long _m, _n, _ld;
  bool _alloc;
  bool _trans;
  T *_data;

 public:
  LocalMatrix()
      : _m(0), _n(0), _ld(0), _alloc(false), _trans(false), _data(NULL) {}

  LocalMatrix(long m, long n)
      : _m(m),
        _n(n),
        _ld(m),
        _alloc(true),
        _trans(false),
        _data(new T[m * n]) {}

  LocalMatrix(long m, long n, T *data)
      : _m(m), _n(n), _ld(m), _alloc(false), _trans(false), _data(data) {}

  LocalMatrix(long m, long n, long ld)
      : _m(m),
        _n(n),
        _ld(ld),
        _alloc(true),
        _trans(false),
        _data(new T[ld * n]) {}

  LocalMatrix(long m, long n, long ld, T *data)
      : _m(m), _n(n), _ld(ld), _alloc(false), _trans(false), _data(data) {}

  ~LocalMatrix() {
    if (_alloc) delete[] _data;
  }

  void override_free() { _alloc = true; }

  long m() const { return _trans ? _n : _m; }

  long n() const { return _trans ? _m : _n; }

  T *data() const { return _data; }

  LocalMatrix<T> *trans() {
    LocalMatrix<T> *C = new LocalMatrix<T>(_m, _n, _ld, _data);
    C->_trans = !_trans;
    return C;
  }

  T &operator()(long i, long j) const {
#ifndef NOCHECK
    assert(i >= 0);
    assert(j >= 0);
    assert(i < m());
    assert(j < n());
#endif
    return _trans ? _data[j + i * _ld] : _data[i + _ld * j];
  }

  T &operator()(long i) const {
#ifndef NOCHECK
    assert((m() == 1 && _trans) || (n() == 1 && !_trans));
#endif
    return _data[i];
  }

  LocalMatrix<T> &operator=(T val) {
    for (long j = 0; j < n(); j++)
      for (long i = 0; i < m(); i++) (*this)(i, j) = val;
    return *this;
  }

  LocalMatrix<T> *operator+(LocalMatrix<T> &B) const {
    LocalMatrix<T> *C = new LocalMatrix<T>(_m, _n, _ld);
    if (_trans) C->trans();
#ifndef NOCHECK
    assert(m() == B.m());
    assert(n() == B.n());
#endif
    for (long j = 0; j < n(); j++)
      for (long i = 0; i < m(); i++) (*C)(i, j) = (*this)(i, j) + B(i, j);
    return C;
  }

  LocalMatrix<T> *operator-(LocalMatrix<T> &B) const {
    LocalMatrix<T> *C = new LocalMatrix<T>(_m, _n, _ld);
    if (_trans) C->trans();
#ifndef NOCHECK
    assert(m() == B.m());
    assert(n() == B.n());
#endif
    for (long j = 0; j < n(); j++)
      for (long i = 0; i < m(); i++) (*C)(i, j) = (*this)(i, j) - B(i, j);
    return C;
  }

  LocalMatrix<T> &operator+=(LocalMatrix<T> &B) {
#ifndef NOCHECK
    assert(m() == B.m());
    assert(n() == B.n());
#endif
    for (long j = 0; j < n(); j++)
      for (long i = 0; i < m(); i++) (*this)(i, j) += B(i, j);
    return *this;
  }

  LocalMatrix<T> &operator-=(LocalMatrix<T> &B) {
#ifndef NOCHECK
    assert(m() == B.m());
    assert(n() == B.n());
#endif
    for (long j = 0; j < n(); j++)
      for (long i = 0; i < m(); i++) (*this)(i, j) -= B(i, j);
    return *this;
  }

  LocalMatrix<T> *operator*(LocalMatrix<T> &B) const {
    LocalMatrix<T> *C;
    T alpha = 1.0, beta = 0.0;

    // check on _view_ dimenions
    assert(n() == B.m());

    C = new LocalMatrix<T>(m(), B.n());

    if (B.n() == 1) {
#ifndef NOCHECK
      assert(!B._trans || (B._trans && B._ld == 1));
#endif
      char transa;
      int m, n, lda;
      int one = 1;
      m = _m;  // true rows of A
      n = _n;  // true cols of A
      lda = _ld;
      transa = _trans ? 'T' : 'N';
      gemv(&transa, &m, &n, &alpha, _data, &lda, B._data, &one, &beta, C->_data,
           &one);
    } else {
      int m, n, k, lda, ldb, ldc;
      char transa, transb;
      m = this->m();  // rows of op( A )
      n = B.n();      // cols of op( B )
      k = this->n();  // cols of op( A )
      lda = _ld;
      ldb = B._ld;
      ldc = C->_ld;
      transa = _trans ? 'T' : 'N';
      transb = B._trans ? 'T' : 'N';
      gemm(&transa, &transb, &m, &n, &k, &alpha, _data, &lda, B._data, &ldb,
           &beta, C->_data, &ldc);
    }

    return C;
  }

};  // end of LocalMatrix

}  // end of namespace cm
