#include "BEM3dLaplace.h"
#include "BEM3dElasticity.h"

#ifdef USE_CUDA
#include "BEM3dLaplaceGPU.h"
extern int ne, nnodes;
#endif


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;


/*
py::array_t<double> make_array(const double* data,
                               ssize_t n)
{
  py::array_t<double> arr(n);
  auto out = arr.mutable_unchecked<1>();
  for (ssize_t i = 0; i < n; ++i)
    out(i) = data[i];
  return arr;
}
*/

py::array_t<double> make_array_rowwise(const double* data,
                                       ssize_t rows,
                                       ssize_t cols)
{
  py::array_t<double> arr({rows, cols});
  auto out = arr.mutable_unchecked<2>();
  for (ssize_t r = 0; r < rows; ++r)
    for (ssize_t c = 0; c < cols; ++c)
      out(r, c) = data[r + c * rows];  // read column-major
  return arr;
}

PYBIND11_MODULE(easybem, m)
{
    m.def("get_laplace3d_mass",
    [](py::array_t<double> nodes,
       py::array_t<int> elements)
    {
        auto nbuf = nodes.request();
        auto ebuf = elements.request();
        size_t nNodes = nbuf.shape[0];
        size_t nElements = ebuf.shape[0];

        double *M = nullptr;
        getLaplace3dMass(
            nNodes,
            static_cast<double*>(nbuf.ptr),
            nElements,
            static_cast<int*>(ebuf.ptr),
            M);

        return make_array_rowwise(M,nElements,nNodes);
    });

    m.def("get_laplace3d_curl",
    [](py::array_t<double> nodes,
       py::array_t<int> elements)
    {
        auto nbuf = nodes.request();
        auto ebuf = elements.request();
        size_t nNodes = nbuf.shape[0];
        size_t nElements = ebuf.shape[0];

        double *C1 = nullptr;
        double *C2 = nullptr;
        double *C3 = nullptr;
        getLaplace3dCurl(
            nNodes,
            static_cast<double*>(nbuf.ptr),
            nElements,
            static_cast<int*>(ebuf.ptr),
            C1,C2,C3);

        return py::make_tuple(
                              make_array_rowwise(C1,nElements,nNodes),
                              make_array_rowwise(C2,nElements,nNodes),
                              make_array_rowwise(C3,nElements,nNodes)
                              );
    });
    
    m.def("get_laplace3d",
    [](py::array_t<double> nodes,
       py::array_t<int> elements,
       int order)
    {
        auto nbuf = nodes.request();
        auto ebuf = elements.request();
        size_t nNodes = nbuf.shape[0];
        size_t nElements = ebuf.shape[0];

        double *V = nullptr;
        double *K = nullptr;
        double *D = nullptr;
        double *M = nullptr;
        getLaplace3d(
            nNodes,
            static_cast<double*>(nbuf.ptr),
            nElements,
            static_cast<int*>(ebuf.ptr),
            V, K, D, M,
            order);

        return py::make_tuple(
            make_array_rowwise(V,nElements,nElements),
            make_array_rowwise(K,nElements,nNodes),
            make_array_rowwise(D,nNodes,nNodes),
            make_array_rowwise(M,nElements,nNodes)
        );
    });

    m.def("eval_laplace3d",
    [](py::array_t<double> nodes,
       py::array_t<int> elements,
       py::array_t<double> points,
       py::array_t<double> neumann,
       py::array_t<double> dirichlet,
       int order)
    {
        auto nbuf = nodes.request();
        auto ebuf = elements.request();
        auto pbuf = points.request();
        auto neumbuf = neumann.request();
        auto dirbuf = dirichlet.request();
        size_t nNodes = nbuf.shape[0];
        size_t nElements = ebuf.shape[0];
        size_t nPoints = pbuf.shape[0];

        double *result = nullptr;
        evaluateLaplace3d(static_cast<double*>(nbuf.ptr),
                          nElements,
                          static_cast<int*>(ebuf.ptr),
                          nPoints,
                          static_cast<double*>(pbuf.ptr),
                          static_cast<double*>(neumbuf.ptr),
                          static_cast<double*>(dirbuf.ptr),
                          result,
                          order);

        return py::make_tuple(
                              make_array_rowwise(result,nPoints,1)
                              );
    });
    
    m.def("get_elasticity3d",
    [](py::array_t<double> nodes,
       py::array_t<int> elements,
       double nu,
       int order)
    {
        auto nbuf = nodes.request();
        auto ebuf = elements.request();
        size_t nNodes = nbuf.shape[0];
        size_t nElements = ebuf.shape[0];

        double *V = nullptr;
        double *K = nullptr;
        double *D = nullptr;
        double *M = nullptr;

        getElasticity3d(
            nNodes,
            static_cast<double*>(nbuf.ptr),
            nElements,
            static_cast<int*>(ebuf.ptr),
            nu,
            V, K, D, M,
            order);

        return py::make_tuple(
            make_array_rowwise(V,3*nElements,3*nElements),
            make_array_rowwise(K,3*nElements,3*nNodes),
            make_array_rowwise(D,3*nNodes,3*nNodes),
            make_array_rowwise(M,3*nElements,3*nNodes)
        );
    });
    

#ifdef USE_CUDA
    m.def("init_gpu",
    [](py::array_t<double> nodes,
       py::array_t<int> elements,
       int order,
       int nblocks, int nthreads)
    {
        auto nbuf = nodes.request();
        auto ebuf = elements.request();
        size_t nNodes = nbuf.shape[0];
        size_t nElements = ebuf.shape[0];

        initGPU(nNodes,
                static_cast<double*>(nbuf.ptr),
                nElements,
                static_cast<int*>(ebuf.ptr),
                order,nblocks,nthreads);
    });

    m.def("close_gpu",
          []()
    {
      closeGPU();
    });

    
    auto apply_gpu = [](py::array_t<double> vec, auto func, int resSize)
                     {
                       auto vbuf = vec.request();
                       py::array_t<double> result(resSize);
                       func(static_cast<double*>(vbuf.ptr),
                            static_cast<double*>(result.request().ptr));
                       return result;
                     };

    m.def("apply_gpu_laplace3d_1layer",
          [&](py::array_t<double> vec)
    {
      return apply_gpu(vec,
                       [](const double* t, double* v)
                       {
                         applyGPULaplace3d1Layer(t, v);

                       },
                       ne);
    });

    m.def("apply_gpu_laplace3d_1layer_linears",
      [&](py::array_t<double> vec)
      {
        return apply_gpu(
                         vec,
                         [](const double* t, double* v) {
                           applyGPULaplace3d1Layer(t, v, 1);
                         },
                         nnodes);
      });

    m.def("apply_gpu_laplace3d_2layer",
          [&](py::array_t<double> vec)
    {
      return apply_gpu(vec, applyGPULaplace3d2Layer, ne);
    });

    m.def("apply_gpu_laplace3d_hypersing",
          [&](py::array_t<double> vec)
    {
      return apply_gpu(vec, applyGPULaplace3dHyperSing, nnodes);
    });
#endif
                        

    m.def("version",
          []()
          {
              return "1.0";
          });
}
