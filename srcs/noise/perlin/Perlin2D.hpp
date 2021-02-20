#ifndef worldgen_noise_Perlin2D_hpp_
#define worldgen_noise_Perlin2D_hpp_

#include "opencl/cl.hpp"
#include "opencl/IRuntime.hpp"

namespace worldgen::noise {
    struct size2_t {
        size_t x;
        size_t y;
    };

    struct pos2_t {
        double x;
        double y;
    };

    struct off2_t {
        double x;
        double y;
    };


    class Perlin2D {
        static constexpr char const * CL_SOURCE_FILE = "res/cl/noise/perlin/perlin2D.cl";
        static constexpr char const * CL_KERNEL_NAME = "fracPerlin2D";

        public:
            using seed_type = unsigned long;

            struct param_t {
                unsigned oct;
                double persistence;
                double lacunarity;
                bool normalize;
            };
        public:
            Perlin2D(seed_type seed, opencl::IRuntime &runtime);
            ~Perlin2D() {}

            std::vector<double> generate(size2_t sz, pos2_t scale = {1., 1.}, off2_t off = {0., 0.}, param_t const & param = {1, 1., 1., false});
            std::vector<double> generate(size2_t sz, double scale, double off, param_t const & param = {1, 1., 1., false});
        private:
            opencl::IRuntime &_runtime;
            cl::Program _prog;

            std::array<int, 512> _perm;
            std::array<cl::double2, 256> _grad;

        private:
            static void init(Perlin2D &obj, seed_type seed);
    };
}

#endif /* end of include guard: worldgen_noise_Perlin2D_hpp_ */
