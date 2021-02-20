#include <iostream>
#include <fstream>
#include <sstream>

#include "opencl/errors.hpp"
#include "opencl/exceptions.hpp"
#include "algo/Normalizer.hpp"

namespace worldgen::algo {
    Normalizer::Normalizer(opencl::IRuntime &rt): _runtime(rt), _minmax(rt) {
        std::ifstream fsources{CL_SOURCE_FILE};

        _sources = std::string(std::istreambuf_iterator<char>(fsources), std::istreambuf_iterator<char>());
    }

    std::vector<double> Normalizer::normalize(std::vector<double> const &data, double min, double max) const {
        std::vector<double> v(data.size(), 0);

        _normalize(data, v, min, max);

        return v;
    }

    std::vector<double> &Normalizer::normalize(std::vector<double> &data, double min, double max) const {
        _normalize(data, data, min, max);

        return data;
    }

    void Normalizer::_normalize(std::vector<double> const &in, std::vector<double> & out, double min, double max) const {
        cl::Context ctx = _runtime.getContext();
        cl::CommandQueue queue = _runtime.getCommandQueue();
        auto [vmin, vmax] = _minmax.reduce(in);
        bound_t from = {vmin, vmax};
        bound_t to = {min, max};

        cl::int1 err;
        cl::Buffer bin(ctx, CL_MEM_READ_ONLY, sizeof(double) * in.size(), NULL, &err);
        cl_throwIfErrorKind(cl::exceptions::ErrorKind::Buffer, err);
        cl::Buffer bout(ctx, CL_MEM_WRITE_ONLY, sizeof(double) * in.size(), NULL, &err);
        cl_throwIfErrorKind(cl::exceptions::ErrorKind::Buffer, err);

        err = queue.enqueueWriteBuffer(bin, CL_FALSE, 0, sizeof(double) * in.size(), in.data());
        cl_throwIfError(err);

        auto kernel = _getKernel();
        //cl::int1 err;
        kernel(cl::EnqueueArgs(queue, cl::NDRange(in.size())), bin, bout, from, to, err);

        if (err != CL_SUCCESS) {
            std::cerr << cl::errors::to_string(err) << " when starting kernel" << std::endl;
            exit(1);
        }

        cl::copy(queue, bout, out.begin(), out.end());
    }

    Normalizer::NormalizeKernel Normalizer::_getKernel() const {
        cl::Program::Sources srcs;
        std::ostringstream preamble;

        preamble << "#define T double" << std::endl;
        preamble << std::endl;

        srcs.push_back(preamble.str() + _sources);


        //std::cerr << preamble.str() + _sources << std::endl << std::endl;

        cl::int1 err;
        cl::Program p{_runtime.getContext(), srcs, &err};
        if (err != CL_SUCCESS) {
            std::cerr << "Error in program creation" << CL_SOURCE_FILE << std::endl;
            std::cerr << cl::errors::to_string(err);
            exit(1);
        }
        p.build();

        for (auto &[_, info] : p.getBuildInfo<CL_PROGRAM_BUILD_LOG>()) {
            std::cerr << info << std::endl;
        }

        NormalizeKernel k(p, CL_KERNEL_NAME, &err);

        if (err != CL_SUCCESS) {
            std::cerr << "Error in kernel creation" << CL_KERNEL_NAME << std::endl;
            std::cerr << cl::errors::to_string(err);
            exit(1);
        }

        return k;
    }
}
