#ifndef opencl_ERRORS_HPP
#define opencl_ERRORS_HPP

#include "opencl/cl.hpp"

namespace cl::errors {
    std::string to_string(cl_int ecode);
}

#endif /* end of include guard: opencl_ERRORS_HPP */
