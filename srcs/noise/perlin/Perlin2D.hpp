#ifndef worldgen_noise_Perlin2D_hpp_
#define worldgen_noise_Perlin2D_hpp_

#include <random>

#include "opencl/cl.hpp"
#include "opencl/IRuntime.hpp"

#include "utils/types.hpp"

namespace worldgen::noise {
    using size2_t = utils::types::size2_t;
    using pos2_t = utils::types::pos2_t;
    using off2_t = utils::types::off2_t;

    class Perlin2D {
        static constexpr char const * CL_SOURCE_FILE = "res/cl/noise/perlin/perlin2D.cl";
        static constexpr char const * KERNEL_NAME_1 = "perlin2D";
        static constexpr char const * KERNEL_NAME_FRAC = "fracPerlin2D";
        static constexpr char const * CL_KERNEL_NAME = "fracPerlin2D";

        public:
            using seed_type = unsigned long;

            struct param_t {
                unsigned oct;
                double persistence;
                double lacunarity;
                bool normalize;
            };

        private:
            struct cl_param_t {
                double weight;
                double freq;
                double amp;
            };

        public:
            Perlin2D(seed_type seed, opencl::IRuntime &runtime);
            ~Perlin2D() {}

            std::vector<double> generate(size2_t sz, pos2_t scale = {1., 1.}, off2_t off = {0., 0.}, param_t const & param = {1, 1., 1., false});
            std::vector<double> generate(size2_t sz, double scale, double off, param_t const & param = {1, 1., 1., false});

        private:
            void _shuffle();
            static double _getWeight(int oct, double amp);
            std::vector<double> _generate_1(size2_t sz, pos2_t scale, off2_t offset);
            std::vector<double> _generate_frac(size2_t sz, pos2_t scale, off2_t offset, param_t const &);
        private:
            opencl::IRuntime &_runtime;
            cl::Program _prog;

            std::array<int, 512> _perm;
            std::array<cl::double2, 256> _grad;

            std::mt19937 _rand;

        private:
            static void init(Perlin2D &obj, seed_type seed);
    };
}

#endif /* end of include guard: worldgen_noise_Perlin2D_hpp_ */
