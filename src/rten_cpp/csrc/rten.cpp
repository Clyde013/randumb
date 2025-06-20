#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <vector>

#include <iostream>

extern "C" {
  /* Creates a dummy empty _C module that can be imported from Python.
     The import from Python will load the .so consisting of this file
     in this extension, so that the TORCH_LIBRARY static initializers
     below are run. */
  PyObject* PyInit__C(void)
  {
      static struct PyModuleDef module_def = {
          PyModuleDef_HEAD_INIT,
          "_C",   /* name of module */
          NULL,   /* module documentation, may be NULL */
          -1,     /* size of per-interpreter state of the module,
                     or -1 if the module keeps state in global variables. */
          NULL,   /* methods */
      };
      return PyModule_Create(&module_def);
  }
}

namespace rten_cpp {

  // CPU implementations go here...
  // IF I HAD ANY

  // Defines the operators
  // schema string docs: https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#func
  TORCH_LIBRARY(rten_cpp, m) {
    m.def("sq5_gen_2d(int M, int N, int seed) -> Tensor");
    m.def("materialize_fwd(Tensor coef, int seed, int P, int Q, int stride_P, int stride_Q) -> Tensor");
  }

}