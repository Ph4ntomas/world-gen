#include <iostream>
#include <fstream>

#include <algorithm>
#include <random>

#include "Perlin2D.hpp"

#include "opencl/exceptions.hpp"
#include "algo/Normalizer.hpp"

namespace worldgen::noise {
    Perlin2D::Perlin2D(seed_type seed, opencl::IRuntime & runtime)
    : _runtime(runtime), _perm(), _grad(), _rand(seed) {
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

        std::vector<double> arr = param.oct > 1 ? _generate_frac(sz, scale, off, param) : _generate_1(sz, scale, off);

        if (param.normalize) {
            algo::Normalizer n(_runtime);

            arr = n.normalize(arr);
        }

        return arr;
    }

    std::vector<double> Perlin2D::_generate_1(size2_t sz, pos2_t scale, off2_t off) {
        auto & ctx = _runtime.getContext();
        auto & queue = _runtime.getCommandQueue();

        std::vector<double> arr(sz.x * sz.y, 0.);

        cl::int1 err;
        cl::Buffer perm(ctx, CL_MEM_READ_ONLY, sizeof(int) * _perm.size(), NULL, &err);
        cl_throwIfErrorKind(cl::exceptions::ErrorKind::Buffer, err);
        cl::Buffer grad(ctx, CL_MEM_READ_ONLY, sizeof(cl::double2) * _grad.size(), NULL, &err);
        cl_throwIfErrorKind(cl::exceptions::ErrorKind::Buffer, err);
        cl::Buffer out(ctx, CL_MEM_READ_WRITE, sizeof(double) * sz.x * sz.y, NULL, &err);
        cl_throwIfErrorKind(cl::exceptions::ErrorKind::Buffer, err);

        cl::KernelFunctor<
            cl::double2, cl::double2,
            cl::Buffer, cl::Buffer, cl::Buffer> perlin2D(_prog, Perlin2D::KERNEL_NAME_1, &err);
        cl_throwIfErrorKind(cl::exceptions::ErrorKind::Kernel, err);

        _shuffle();

        err = queue.enqueueWriteBuffer(perm, CL_FALSE, 0, sizeof(int) * _perm.size(), _perm.data());
        cl_throwIfError(err);
        err = queue.enqueueWriteBuffer(grad, CL_FALSE, 0, sizeof(cl::double2) * _grad.size(), _grad.data());
        cl_throwIfError(err);

        perlin2D(
            cl::EnqueueArgs(cl::NDRange(sz.x, sz.y)),
            {scale.x, scale.y}, {off.x, off.y},
            perm, grad, out,
            err
        );
        cl_throwIfErrorKind(cl::exceptions::ErrorKind::Kernel, err);

        err = cl::copy(out, arr.begin(), arr.end());
        cl_throwIfError(err);

        return arr;
    }

    std::vector<double> Perlin2D::_generate_frac(size2_t sz, pos2_t scale, off2_t off, param_t const &p) {
        auto & ctx = _runtime.getContext();
        auto & queue = _runtime.getCommandQueue();

        std::vector<double> arr(sz.x * sz.y, 0.);
        double weight = _getWeight(p.oct, p.persistence);

        cl::int1 err;
        cl::Buffer perm(ctx, CL_MEM_READ_ONLY, sizeof(int) * _perm.size(), NULL, &err);
        cl_throwIfErrorKind(cl::exceptions::ErrorKind::Buffer, err);
        cl::Buffer grad(ctx, CL_MEM_READ_ONLY, sizeof(cl::double2) * _grad.size(), NULL, &err);
        cl_throwIfErrorKind(cl::exceptions::ErrorKind::Buffer, err);
        cl::Buffer out(ctx, CL_MEM_READ_WRITE, sizeof(double) * sz.x * sz.y, NULL, &err);
        cl_throwIfErrorKind(cl::exceptions::ErrorKind::Buffer, err);

        err = queue.enqueueWriteBuffer(out, CL_FALSE, 0, sizeof(double) * arr.size(), arr.data());
        cl_throwIfError(err);

        cl::KernelFunctor<
            cl_param_t,
            cl::double2, cl::double2,
            cl::Buffer, cl::Buffer, cl::Buffer> perlin2D(_prog, Perlin2D::KERNEL_NAME_FRAC, &err);
        cl_throwIfErrorKind(cl::exceptions::ErrorKind::Kernel, err);

        for (int i = 0; i < p.oct; ++i) {
            double freq = std::pow(p.lacunarity, i);
            double amp = std::pow(p.persistence, i);

            _shuffle();

            err = queue.enqueueWriteBuffer(perm, CL_FALSE, 0, sizeof(int) * _perm.size(), _perm.data());
            cl_throwIfError(err);
            err = queue.enqueueWriteBuffer(grad, CL_FALSE, 0, sizeof(cl::double2) * _grad.size(), _grad.data());
            cl_throwIfError(err);

            perlin2D(
                    cl::EnqueueArgs(cl::NDRange(sz.x, sz.y)),
                    {weight, freq, amp},
                    {scale.x, scale.y}, {off.x, off.y},
                    perm, grad, out,
                    err
                    );
            cl_throwIfErrorKind(cl::exceptions::ErrorKind::Kernel, err);
        }

        err = cl::copy(out, arr.begin(), arr.end());
        cl_throwIfError(err);

        return arr;
    }

    void Perlin2D::init(Perlin2D &obj, seed_type seed) {
        std::uniform_int_distribution<std::size_t> d_i(0, 256);
        std::uniform_real_distribution<double> d_r;

        auto &g = obj._rand;

        std::iota(obj._perm.begin(), obj._perm.begin() + 256, 0);

        for (size_t i = 0; i < 256; ++i) {
            double t = 2. * M_PI * d_r(g);
            double u = d_r(g) + d_r(g);
            double r = u > 1 ? 2. - u : u;

            obj._grad[i] = { r * std::cos(t), r * std::sin(t) };
        }
    }

    void Perlin2D::_shuffle() {
        std::shuffle(_grad.begin(), _grad.end(), _rand);
        std::shuffle(_perm.begin(), _perm.begin() + 256, _rand);

        std::copy(
                _perm.begin(), _perm.begin() + 256,
                _perm.begin() + 256);
    }

    double Perlin2D::_getWeight(int oct, double pers) {
        double amp = 1.;
        double weight = 0;

        for (int i = 0; i < oct; ++i) {
            weight += amp;
            amp *= pers;
        }

        return weight;
    }
}
