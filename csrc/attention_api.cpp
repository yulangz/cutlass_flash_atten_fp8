#include <torch/extension.h>
#include <torch/python.h>

#include "attention_api.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // m.def("package_name", &function_name, "function_docstring"")
    m.def("_flash_attention_v2_fp8", &flash_attention_v2_fp8,
          "Flash attention v2 implement in cutlass");
}
