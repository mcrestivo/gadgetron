#pragma once

#include "cuNDArray.h"
#include "matrixOperator.h"
#include "vector_td_utilities.h"
#include "solvers_export.h"

#include <boost/smart_ptr.hpp>
#include <vector>

template <class REAL, class T, unsigned int D> class cuVariableGaussOperator : public matrixOperator<REAL, cuNDArray<T> >
{

 public:
  cuVariableGaussOperator() : matrixOperator<REAL, cuNDArray<T> >(){}

  virtual ~cuVariableGaussOperator() {}
  void set_sigma(cuNDArray<REAL> * sigma);
  
  virtual int mult_MH_M( cuNDArray<T> *in, cuNDArray<T> *out, bool accumulate = false );
  virtual int mult_MH( cuNDArray<T> *in, cuNDArray<T> *out, bool accumulate = false );
  virtual int mult_M( cuNDArray<T> *in, cuNDArray<T> *out, bool accumulate = false );


 protected:
  cuNDArray<REAL>* _sigma;
  boost::shared_ptr<cuNDArray<REAL> > _norm;


};