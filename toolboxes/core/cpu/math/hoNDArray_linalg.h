
#pragma once

#include "cpucore_math_export.h"
#include "hoMatrix.h"

#ifdef USE_ARMADILLO
    #include "hoArmadillo.h"
#endif // USE_ARMADILLO

#ifndef lapack_int
    #define lapack_int int
#endif // lapack_int

/// ----------------------------------------------------------------------
/// the fortran interface of lapack and blas functions are called
/// ----------------------------------------------------------------------

namespace Gadgetron
{

// following matrix computation calls lapacke functions

/// C = A*B for complex float
EXPORTCPUCOREMATH bool gemm(hoNDArray< std::complex<float> >& C, const hoNDArray< std::complex<float> >& A, const hoNDArray< std::complex<float> >& B);
/// if transA==true, C = A'*B
/// if transB==true, C=A*B'
/// if both are true, C=A'*B'
template<typename T> EXPORTCPUCOREMATH
bool gemm(hoNDArray<T>& C, 
        const hoNDArray<T>& A, bool transA, 
        const hoNDArray<T>& B, bool transB);

/// perform a symmetric rank-k update (no conjugated).
template<typename T> EXPORTCPUCOREMATH 
bool syrk(hoNDArray<T>& C, const hoNDArray<T>& A, char uplo, bool isATA);

/// perform a Hermitian rank-k update.
template<typename T> EXPORTCPUCOREMATH 
bool herk(hoNDArray<T>& C, const hoNDArray<T>& A, char uplo, bool isAHA);

/// compute the Cholesky factorization of a real symmetric positive definite matrix A
template<typename T> EXPORTCPUCOREMATH 
bool potrf(hoMatrix<T>& A, char uplo);

/// compute all eigenvalues and eigenvectors of a Hermitian matrix A
template<typename T> EXPORTCPUCOREMATH 
bool heev(hoMatrix<T>& A, hoMatrix<typename realType<T>::Type>& eigenValue);

template<typename T> EXPORTCPUCOREMATH
bool heev(hoMatrix< std::complex<T> >& A, hoMatrix<  std::complex<T> >& eigenValue);

/// compute inverse of a symmetric (Hermitian) positive-definite matrix A
template<typename T> EXPORTCPUCOREMATH 
bool potri(hoMatrix<T>& A);

/// compute the inverse of a triangular matrix A
template<typename T> EXPORTCPUCOREMATH 
bool trtri(hoMatrix<T>& A, char uplo);

/// solve Ax=b, a symmetric or Hermitian positive-definite matrix A and multiple right-hand sides b
/// b is replaced with x
template<typename T> EXPORTCPUCOREMATH 
bool posv(hoMatrix<T>& A, hoMatrix<T>& b);

/// solve Ax=b with Tikhonov regularization
template<typename T> EXPORTCPUCOREMATH
bool SolveLinearSystem_Tikhonov(hoMatrix<T>& A, hoMatrix<T>& b, hoMatrix<T>& x, double lamda);

/// Computes the LU factorization of a general m-by-n matrix
/// this function is called by general matrix inversion
template<typename T> EXPORTCPUCOREMATH 
bool getrf(hoMatrix<T>& A, hoNDArray<lapack_int>& ipiv);

/// Computes the inverse of an LU-factored general matrix
template<typename T> EXPORTCPUCOREMATH 
bool getri(hoMatrix<T>& A);

}