#ifndef opencl_DEFAULTRUNTIME_HPP
#define opencl_DEFAULTRUNTIME_HPP

#include "opencl/cl.hpp"
#include "opencl/AbstractRuntime.hpp"

namespace worldgen::opencl {
    class DefaultRuntime: public AbstractRuntime {
        public:
            DefaultRuntime();
            virtual ~DefaultRuntime() {}
    };
}

#endif /* end of include guard: opencl_ABSTRACTRUNTIME_HPP */
