#include <functional>

#include <fstream>
#include <iostream>
#include <sstream>

#include "opencl/exceptions.hpp"
#include "landmass/Landmass.hpp"

namespace worldgen::landmass {
    Landmass::Landmass(seed_type rand_seed, opencl::IRuntime &runtime, param_t rand_params)
        :_runtime(runtime),
        _rand_gen(rand_seed), _rand_params(rand_params)
    {
        reseed();
        std::ifstream fs(CL_SOURCE_FILE);

        _source = std::string(std::istreambuf_iterator<char>(fs), std::istreambuf_iterator<char>());
    }

    Landmass::~Landmass() {}

    std::vector<double> Landmass::generate(size2_t sz, std::vector<double> const &fm) const {
        std::vector<double> ret(sz.x * sz.y);

        _generate(sz, fm, ret);
        return ret;
    }

    std::vector<double> & Landmass::generate(size2_t sz, std::vector<double> &fm) const {
        _generate(sz, fm, fm);

        return fm;
    }

    void Landmass::_generate(
        size2_t sz,
        std::vector<double> const & in, std::vector<double> &out
    ) const {
        cl::Context ctx = _runtime.getContext();
        cl::CommandQueue queue = _runtime.getCommandQueue();
        out.reserve(sz.x * sz.y);

        cl::int1 err;
        cl::Buffer bin(ctx, CL_MEM_READ_ONLY, sizeof(double) * in.size(), NULL, &err);
        cl_throwIfErrorKind(cl::exceptions::ErrorKind::Buffer, err);
        cl::Buffer bout(ctx, CL_MEM_WRITE_ONLY, sizeof(double) * in.size(), NULL, &err);
        cl_throwIfErrorKind(cl::exceptions::ErrorKind::Buffer, err);
        cl::Buffer seeds(ctx, CL_MEM_READ_ONLY, sizeof(seed_t) * _seeds.size(), NULL, &err);
        cl_throwIfErrorKind(cl::exceptions::ErrorKind::Buffer, err);

        err = queue.enqueueWriteBuffer(bin, CL_FALSE, 0, sizeof(double) * in.size(), in.data());
        cl_throwIfError(err);
        err = queue.enqueueWriteBuffer(bin, CL_FALSE, 0, sizeof(seed_t) * _seeds.size(), _seeds.data());
        cl_throwIfError(err);

        auto kernel = _getKernel();
        kernel(
            cl::EnqueueArgs(queue, cl::NDRange(sz.x, sz.y)),
            _rand_params.global_threshold,
            bin, bout, seeds,
            err
        );

        if (err != CL_SUCCESS) {
            std::cerr << cl::errors::to_string(err) << " when starting kernel" << std::endl;
            exit(1);
        }

        cl::copy(queue, bout, out.begin(), out.end());
    }

    Landmass & Landmass::reseed(std::optional<seed_type> new_seed) {
        using IntDice = std::uniform_int_distribution<unsigned int>;
        using RealDice = std::uniform_real_distribution<double>;

        //[[unlikely]] // This is a c++20 feature. move it to a macro ?
        if (new_seed)
            _rand_gen.seed(new_seed.value());

        auto idice = std::bind(IntDice(), _rand_gen, std::placeholders::_1);
        auto rdice = std::bind(RealDice(), _rand_gen, std::placeholders::_1);

        unsigned int n_pos = idice(IntDice::param_type{_rand_params.min_pos, _rand_params.max_pos});
        unsigned int n_neg = idice(IntDice::param_type{_rand_params.min_neg, _rand_params.max_neg});

        std::vector<SeedType> s_types;
        std::fill_n(std::back_inserter(s_types), n_pos, SeedType::Positive);
        std::fill_n(std::back_inserter(s_types), n_neg, SeedType::Negative);
        std::shuffle(s_types.begin(), s_types.end(), _rand_gen); // not useful ?

        _seeds.clear();
        std::transform(s_types.begin(), s_types.end(), std::back_inserter(_seeds), [&] (SeedType s) {
            return seed_t{
                s,
                {
                    rdice(RealDice::param_type{0., double(_rand_params.size.x)}),
                    rdice(RealDice::param_type{0., double(_rand_params.size.y)})
                },
                s ? _rand_params.pos_threshold : _rand_params.neg_threshold,
                s ? _rand_params.pos_off : _rand_params.neg_offset
            };
        });

        return *this;
    }

    Landmass::LandmassKernel Landmass::_getKernel() const {
        cl::Program::Sources srcs;
        std::ostringstream preamble, comp_options;

        preamble << "#define RType double" << std::endl;
        preamble << "#define NB_SEED " << _seeds.size() << std::endl;
        preamble << std::endl;

        srcs.push_back(preamble.str() + _source);

        //std::cerr << "--------- preamble: ---------" << std::endl;
        //std::cerr << preamble.str() << std::endl << std::endl;
        //std::cerr << "Building program with the following sources " << std::endl;
        //std::cerr << preamble.str() + _sources << std::endl;
        //std::cerr << std::endl;

        comp_options << "-I" << CL_INCLUDE_DIR << std::endl;

        cl_int err;
        cl::Program p{_runtime.getContext(), srcs, &err};
        if (err != CL_SUCCESS) {
            std::cerr << "Error in program creation" << CL_SOURCE_FILE << std::endl;
            std::cerr << cl::errors::to_string(err);
            exit(1);
        }
        p.build(comp_options.str().c_str());

        for (auto &[_, info] : p.getBuildInfo<CL_PROGRAM_BUILD_LOG>()) {
            std::cerr << info << std::endl;
        }

        auto k = LandmassKernel(p, CL_KERNEL_NAME, &err);

        if (err != CL_SUCCESS) {
            std::cerr << "Error in kernel creation" << CL_KERNEL_NAME << std::endl;
            std::cerr << cl::errors::to_string(err);
            exit(1);
        }

        return k;
    }
}
