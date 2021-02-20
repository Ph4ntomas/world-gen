#ifndef worldgen_algo_Reducer_hpp
#define worldgen_algo_Reducer_hpp

#include "opencl/IRuntime.hpp"

namespace worldgen::algo {
    namespace reduce_op {
        struct min_t {};
        struct max_t {};
        struct sum_t {};

        static constexpr min_t min;
        static constexpr max_t max;
        static constexpr sum_t sum;
    }

    class Reducer {
        static constexpr char const * CL_SOURCE_FILE = "res/cl/algo/reduce.cl";
        static constexpr char const * CL_KERNEL_NAME = "reduce";
        static constexpr unsigned int const MAX_THREADS = 128;
        static constexpr unsigned int const MAX_BLOCK = 64;

        using ReduceKernel = cl::KernelFunctor<cl::Buffer, cl::Buffer, unsigned int>;

        public:
            template <typename OpTag>
            Reducer(OpTag, opencl::IRuntime &runtime);
            double reduce(std::vector<double> const &);

        private:
            Reducer(std::string const &reduce_op, double init, opencl::IRuntime &runtime);

            void getNumThreadAndBlock(unsigned int size, unsigned int &thread, unsigned int &block) const;
            ReduceKernel getReduceKernel(unsigned int thread, bool isPow2);

        private:
            opencl::IRuntime &_runtime;
            std::string _preamble;
            std::string _sources;
    };

    template <> Reducer::Reducer(reduce_op::min_t, opencl::IRuntime &runtime);
    template <> Reducer::Reducer(reduce_op::max_t, opencl::IRuntime &runtime);
    template <> Reducer::Reducer(reduce_op::sum_t, opencl::IRuntime &runtime);
}


#endif /* end of include guard: worldgen_algo_Reducer_hpp */
