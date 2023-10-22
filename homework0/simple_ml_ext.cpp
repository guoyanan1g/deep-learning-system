#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

float* matmul(const float *x,const float* y,int m,int n,int k){
  //x:[m,n] y:[n,k]
  float* z=new float[m*k];
  memset(z,0,m*k*sizeof(float));
  for(int i=0;i<m;i++)//对于x的每一行
    for(int j=0;j<k;j++){//遍历y的每一列
      for(int l=0;l<n;l++)
        z[i*k+j]+=x[i*n+l]*y[l*k+j];
    }
  return z;
}
float *slice_x(const float* x,int start,int end,int n){
  float* z=new float[(end-start+1)*n];
  int l=0;
  for(int i=start;i<=end;i++){
    for(int j=0;j<n;j++)
      z[l*n+j]=x[i*n+j];
    l++;
  }
    
  return z;
}

unsigned char* slice_y(const unsigned char* y,int start,int end){
  unsigned char* z=new unsigned char[end-start+1];
  for(int i=start;i<=end;i++){
    z[i-start]=y[i];
  }
  return z;
}

float *transpose(float *x,int m,int n){
  float *z=new float[m*n];
  for(int i=0;i<m;i++)
    for(int j=0;j<n;j++){
      z[n*j+i]=x[i*m+j];
      //建议画图对照
    }
    return z;
}

float *softmax(const float *x,int m,int k){
  float *z=new float[m*k];
  for(int i=0;i<m;i++)
    for(int j=0;j<k;j++)
      z[i*k+j]=exp(x[i*k+j]);
  for(int i=0;i<m;i++){
    float sum=0;
    for(int j=0;j<k;j++)sum+=z[i*k+j];
    for(int j=0;j<k;j++)z[i*k+j]/=sum;
  }
  return z;
}

void minus(float *x,float *y,int m,int n){
  for(int i=0;i<m*n;i++)
    x[i]=x[i]-y[i];
}
void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    size_t times=m/batch;
    for(size_t i=0;i<times;i++){
      size_t start=i*batch;
      size_t end=std::min(m,start+batch-1);
      if(end==m)break;
      float *z;
      float *x;
      unsigned char *yy;
      x=slice_x(X,start,end,n);
      yy=slice_y(y,start,end);
      z=matmul(x,theta,batch,n,k);
      z=softmax(z,batch,k);

      size_t *num_y=new size_t[batch];
      for(size_t i=0;i<batch;i++)
        num_y[i]=yy[i]-'0';

      for(size_t i=0;i<batch;i++){
        z[i*k+num_y[i]]-=1;
      }
      x=transpose(x,batch,n);
      float *grad;
      grad=matmul(x,z,n,batch,k);
      for(size_t i=0;i<n*k;i++)
        grad[i]*=(lr/batch);
      minus(theta,grad,n,k);
      delete[] z;
      delete[] x;
      delete[] yy;
      delete[] grad;
    }

    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
