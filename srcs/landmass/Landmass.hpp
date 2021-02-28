#ifndef worldgen_landmass_Landmass_hpp
#define worldgen_landmass_Landmass_hpp

#include <optional>
#include <random>
#include <vector>

#include "opencl/cl.hpp"
#include "opencl/IRuntime.hpp"

#include "utils/types/size2_t.hpp"

namespace worldgen::landmass {
    using size2_t = utils::types::size2_t;
    class Landmass {
        public:
            using seed_type = unsigned long;
            struct param_t {
                size2_t size; //map total size

                unsigned int min_pos; // min number of positives seeds
                unsigned int max_pos; // max number of positives seeds
                double pos_off;
                double pos_threshold; //max dist of action for positives seeds

                unsigned int min_neg;
                unsigned int max_neg;
                double neg_offset;
                double neg_threshold; //max dist of action for negatives seeds

                double global_threshold; // above this value
            };

        private:
            enum struct SeedType : int {
                Negative = 0,
                Positive = 1
            };

            struct seed_t {
                SeedType type;
                cl::double2 pos;
                double range;
                double offset;
            };

            using LandmassKernel = cl::KernelFunctor<double, cl::Buffer, cl::Buffer, cl::Buffer>;
        public:
            Landmass(seed_type seed, opencl::IRuntime &runtime, param_t params);
            ~Landmass();

            std::vector<double> generate(size2_t sz, std::vector<double> const &flatmap) const;
            std::vector<double> &generate(size2_t sz, std::vector<double> &flatmap) const;

            Landmass &reseed(std::optional<seed_type> new_seed = std::nullopt);

        private:
            [[nodiscard]]
            LandmassKernel _getKernel() const;

            void _generate(size2_t sz, std::vector<double> const & data_in, std::vector<double> &data_out) const;
        private:
            opencl::IRuntime &_runtime;
            std::mt19937 _rand_gen;
            param_t _rand_params;

            std::vector<seed_t> _seeds;

            std::string _source;

            static constexpr char const * CL_INCLUDE_DIR = "res/cl";

            static constexpr char const * CL_SOURCE_FILE = "res/cl/landmass/landmass.cl";
            static constexpr char const * CL_KERNEL_NAME = "landmass";
    };
}


#endif /* end of include guard: worldgen_landmass_Landmass_hpp */
