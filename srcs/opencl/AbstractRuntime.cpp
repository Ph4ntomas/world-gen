#include "opencl/AbstractRuntime.hpp"

namespace worldgen::opencl {
    AbstractRuntime::AbstractRuntime(cl::Platform p, cl::Device d, cl::Context ctx, cl::CommandQueue queue)
    : _plat(p), _ctx(ctx), _dev(d), _queue(queue) {

    }

    [[nodiscard]]
    cl::Platform const & AbstractRuntime::getPlatform() const noexcept {
        return _plat;
    }
    [[nodiscard]]
    cl::Platform & AbstractRuntime::getPlatform() noexcept {
        return _plat;
    }

    [[nodiscard]]
    cl::Context const & AbstractRuntime::getContext() const noexcept {
        return _ctx;
    }
    [[nodiscard]]
    cl::Context & AbstractRuntime::getContext() noexcept {
        return _ctx;
    }

    [[nodiscard]]
    cl::Device const & AbstractRuntime::getDevice() const noexcept {
        return _dev;
    }
    [[nodiscard]]
    cl::Device & AbstractRuntime::getDevice() noexcept {
        return _dev;
    }

    [[nodiscard]]
    cl::CommandQueue const & AbstractRuntime::getCommandQueue() const noexcept {
    return _queue;
    }

    [[nodiscard]]
    cl::CommandQueue & AbstractRuntime::getCommandQueue() noexcept {
    return _queue;
    }
}
