#include "Perlin2D.hpp"

#include <iostream>
#include <fstream>
#include <random>

#include "opencl/exceptions.hpp"
#include "algo/Normalizer.hpp"

namespace worldgen::noise {
    Perlin2D::Perlin2D(seed_type seed, opencl::IRuntime & runtime)
    : _runtime(runtime), _perm(), _grad() {
        Perlin2D::init(*this, seed);

        cl::Program::Sources srcs;

        std::ifstream srcs_file(CL_SOURCE_FILE);
        std::string const kern{std::istreambuf_iterator<char>(srcs_file), std::istreambuf_iterator<char>()};

        srcs.push_back(kern);

        cl_int ep;
        _prog = cl::Program(_runtime.getContext(), srcs, &ep);
        _prog.build();

        for (auto &[_, info] :_prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>()) {
            std::cerr << info << std::endl;
        }
    }

    std::vector<double> Perlin2D::generate(size2_t sz, double scale, double off, param_t const &param) {
        return generate(sz, {scale, scale}, {off, off}, param);
    }

    std::vector<double> Perlin2D::generate(size2_t sz, pos2_t scale, off2_t off, param_t const &param) {
        auto & ctx = _runtime.getContext();
        auto & queue = _runtime.getCommandQueue();

        std::vector<double> arr(sz.x * sz.y, 0.);
        {
            cl::int1 err;
            cl::Buffer perm(ctx, CL_MEM_READ_ONLY, sizeof(int) * _perm.size(), NULL, &err);
            cl_throwIfErrorKind(cl::exceptions::ErrorKind::Buffer, err);
            cl::Buffer grad(ctx, CL_MEM_READ_ONLY, sizeof(cl::double2) * _grad.size(), NULL, &err);
            cl_throwIfErrorKind(cl::exceptions::ErrorKind::Buffer, err);
            cl::Buffer out(ctx, CL_MEM_WRITE_ONLY, sizeof(double) * sz.x * sz.y, NULL, &err);
            cl_throwIfErrorKind(cl::exceptions::ErrorKind::Buffer, err);

            err = queue.enqueueWriteBuffer(perm, CL_TRUE, 0, sizeof(int) * _perm.size(), _perm.data());
            cl_throwIfError(err);
            err = queue.enqueueWriteBuffer(grad, CL_TRUE, 0, sizeof(cl::double2) * _grad.size(), _grad.data());
            cl_throwIfError(err);

            cl::KernelFunctor<
                param_t,
                cl::double2, cl::double2,
                cl::Buffer, cl::Buffer, cl::Buffer> perlin2D(_prog, Perlin2D::CL_KERNEL_NAME, &err);

            cl_throwIfErrorKind(cl::exceptions::ErrorKind::Kernel, err);
            perlin2D(
                    cl::EnqueueArgs(cl::NDRange(sz.x, sz.y)),
                    param,
                    {scale.x, scale.y}, {off.x, off.y},
                    perm, grad, out,
                    err
                    );

            cl_throwIfErrorKind(cl::exceptions::ErrorKind::Kernel, err);


            err = cl::copy(out, arr.begin(), arr.end());
            cl_throwIfError(err);
        }
        if (param.normalize) {
            algo::Normalizer n(_runtime);

            arr = n.normalize(arr);
        }

        return arr;
    }

    void Perlin2D::init(Perlin2D &obj, seed_type seed) {
        std::uniform_int_distribution<std::size_t> d_i(0, 256);
        std::uniform_real_distribution<double> d_r;

        std::mt19937 g(seed);

        std::iota(obj._perm.begin(), obj._perm.begin() + 256, 0);

        for (size_t i = 0; i < 256; ++i) {
            std::swap(obj._perm[i], obj._perm[d_i(g)]);
            obj._perm [i + 256] = obj._perm[i];

            double t = 2. * M_PI * d_r(g);
            double u = d_r(g) + d_r(g);
            double r = u > 1 ? 2. - u : u;

            obj._grad[i] = { r * std::cos(t), r * std::sin(t) };
        }
    }
}
