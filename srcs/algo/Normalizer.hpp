#ifndef worldgen_algo_Normalizer_hpp
#define worldgen_algo_Normalizer_hpp

#include "opencl/IRuntime.hpp"
#include "algo/MinMax.hpp"

namespace worldgen::algo {
    class Normalizer {
        static constexpr char const * CL_SOURCE_FILE = "res/cl/algo/normalize.cl";
        static constexpr char const * CL_KERNEL_NAME = "normalize";

        struct bound_t {
            double vmin;
            double vmax;
        };
        using NormalizeKernel = cl::KernelFunctor<cl::Buffer, cl::Buffer, bound_t, bound_t>;
        public:
            Normalizer(opencl::IRuntime &runtime);

            std::vector<double> normalize(std::vector<double> const &data, double min = 0., double max = 1.) const;
            std::vector<double> &normalize(std::vector<double> &data, double min = 0., double max = 1.) const;

        private:
            void _normalize(std::vector<double> const &in, std::vector<double> &out, double min, double max) const;
            NormalizeKernel _getKernel() const;

        private:
            opencl::IRuntime &_runtime;
            MinMax _minmax;
            std::string _sources;
    };
}

#endif /* end of include guard: worldgen_algo_Normalizer_hpp */
