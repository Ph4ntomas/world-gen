#ifndef opencl_ABSTRACTRUNTIME_HPP
#define opencl_ABSTRACTRUNTIME_HPP

#include "opencl/cl.hpp"
#include "opencl/IRuntime.hpp"

namespace worldgen::opencl {
    class AbstractRuntime: public IRuntime {
        public:
            AbstractRuntime(cl::Platform platform, cl::Device dev, cl::Context ctx, cl::CommandQueue queue);
            virtual ~AbstractRuntime() {}

            [[nodiscard]]
            cl::Platform const & getPlatform() const noexcept;
            [[nodiscard]]
            cl::Platform & getPlatform() noexcept;

            [[nodiscard]]
            cl::Context const & getContext() const noexcept;
            [[nodiscard]]
            cl::Context & getContext() noexcept;

            [[nodiscard]]
            cl::Device const & getDevice() const noexcept;
            [[nodiscard]]
            cl::Device & getDevice() noexcept;

            [[nodiscard]]
            cl::CommandQueue const & getCommandQueue() const noexcept;
            [[nodiscard]]
            cl::CommandQueue & getCommandQueue() noexcept;

        protected:
            cl::Platform _plat;
            cl::Context _ctx;
            cl::Device _dev;
            cl::CommandQueue _queue;
    };
}

#endif /* end of include guard: opencl_ABSTRACTRUNTIME_HPP */
