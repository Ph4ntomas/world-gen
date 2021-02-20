#ifndef opencl_IRUNTIME_HPP_
#define opencl_IRUNTIME_HPP_

#include "opencl/cl.hpp"

namespace worldgen::opencl {
    class IRuntime {
        public:
            virtual ~IRuntime() {}

            [[nodiscard]]
            virtual cl::Platform const & getPlatform() const noexcept = 0;
            [[nodiscard]]
            virtual cl::Platform & getPlatform() noexcept = 0;

            [[nodiscard]]
            virtual cl::Context const & getContext() const noexcept = 0;
            [[nodiscard]]
            virtual cl::Context & getContext() noexcept = 0;

            [[nodiscard]]
            virtual cl::Device const & getDevice() const noexcept = 0;
            [[nodiscard]]
            virtual cl::Device & getDevice() noexcept = 0;

            [[nodiscard]]
            virtual cl::CommandQueue const & getCommandQueue() const noexcept = 0;
            [[nodiscard]]
            virtual cl::CommandQueue & getCommandQueue() noexcept = 0;
    };
}

#endif /* end of include guard: opencl_IRUNTIME_HPP_ */
