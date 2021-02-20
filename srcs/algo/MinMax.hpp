#ifndef worldgen_algo_MinMax_hpp
#define worldgen_algo_MinMax_hpp

#include <vector>

#include "opencl/IRuntime.hpp"

namespace worldgen::algo {
    class MinMax {
        static constexpr char const * CL_SOURCE_FILE = "res/cl/algo/minmax.cl";
        static constexpr char const * CL_KERNEL_NAME = "reduce_minmax";
        static constexpr char const * CL_KERNEL_SUB_NAME = "reduce_minmax_sub";
        static constexpr unsigned int const MAX_THREADS = 128;
        static constexpr unsigned int const MAX_BLOCK = 64;

        using ReduceKernel = cl::KernelFunctor<cl::Buffer, cl::Buffer, unsigned int>;

        public:
            MinMax(opencl::IRuntime &runtime);
            std::pair<double, double> reduce(std::vector<double> const &) const;

        private:

            void getNumThreadAndBlock(unsigned int size, unsigned int &thread, unsigned int &block) const;
            ReduceKernel getReduceKernel(unsigned int thread, bool isPow2) const;
            ReduceKernel getReduceSubKernel(unsigned int thread, bool isPow2) const;

            cl::Program getProgram(unsigned int block_size, bool isPow2) const;

        private:
            opencl::IRuntime &_runtime;
            std::string _sources;
    };
}

#endif /* end of include guard: worldgen_algo_MinMax_hpp */
