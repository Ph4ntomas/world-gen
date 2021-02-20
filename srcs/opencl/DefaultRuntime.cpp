#include "./DefaultRuntime.hpp"

namespace worldgen::opencl {
    DefaultRuntime::DefaultRuntime():
        AbstractRuntime(
            cl::Platform::getDefault(),
            cl::Device::getDefault(),
            cl::Context::getDefault(),
            cl::CommandQueue::getDefault()
        )
    {}
}
