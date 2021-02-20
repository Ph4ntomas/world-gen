#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "opencl/exceptions.hpp"
#include "opencl/errors.hpp"
#include "algo/MinMax.hpp"

static bool isPow2(unsigned x) {
    return !(x & (x - 1));
}

static unsigned nextPow2(unsigned x) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

namespace worldgen::algo {
    MinMax::MinMax(opencl::IRuntime &rt): _runtime(rt) {
        std::ifstream src(CL_SOURCE_FILE);
        _sources = std::string(std::istreambuf_iterator<char>(src), std::istreambuf_iterator<char>());
    }

    std::pair<double, double> MinMax::reduce(std::vector<double> const &data) const {
        auto & ctx = _runtime.getContext();
        auto & queue = _runtime.getCommandQueue();

        unsigned int nThread = 0;
        unsigned int nBlock = 0;

        cl::int1 err;

        getNumThreadAndBlock(data.size(), nThread, nBlock);

        cl::Buffer in(ctx, CL_MEM_READ_ONLY, sizeof(double) * data.size(), NULL, &err);
        cl_throwIfErrorKind(cl::exceptions::ErrorKind::Buffer, err);
        cl::Buffer out(ctx, CL_MEM_READ_WRITE, sizeof(cl::double2) * nBlock, NULL, &err);
        cl_throwIfErrorKind(cl::exceptions::ErrorKind::Buffer, err);

        cl_throwIfError(queue.enqueueWriteBuffer(in, CL_FALSE, 0, sizeof(double) * data.size(), data.data()));

        auto k = getReduceKernel(nThread, isPow2(data.size()));
        k.getKernel().setArg(3, sizeof(cl::double2) * nThread, NULL);
        k(cl::EnqueueArgs(queue, cl::NDRange(nBlock * nThread), cl::NDRange(nThread)),
            in, out, data.size()
        );

        unsigned int block = 0;
        unsigned int thread = 0;

        for (unsigned s = nBlock; s > 1; s = (s + thread * 2 - 1) / (thread * 2)) {
            //std::cerr << s << std::endl;
            getNumThreadAndBlock(s, thread, block);
            auto k = getReduceSubKernel(thread, isPow2(s));
            k.getKernel().setArg(3, sizeof(cl::double2) * thread, NULL);
            k(cl::EnqueueArgs(queue, cl::NDRange(block * thread), cl::NDRange(thread)),
                out, out, s
            );
        }

        cl::double2 ret;
        cl_throwIfError(queue.enqueueReadBuffer(out, CL_TRUE, 0, sizeof(cl::double2), &ret));
        return { ret.s[0], ret.s[1] };
    }

    void MinMax::getNumThreadAndBlock(unsigned int size, unsigned int &thread, unsigned int &block) const {
        thread = (size < MAX_THREADS) ? nextPow2((size + 1) / 2) : MAX_THREADS;
        block = (size + thread * 2 - 1) / (thread * 2);
        block = std::min(MAX_BLOCK, block);
    }

    cl::Program MinMax::getProgram(unsigned int block_size, bool isPow2) const {
        cl::Program::Sources srcs;
        std::ostringstream preamble;

        preamble << "#define T double" << std::endl;
        preamble << "#define T2 double2" << std::endl;
        preamble << "#define IS_SZ_POW2 " << isPow2 << std::endl;
        preamble << "#define BLOCK_SIZE " << block_size << std::endl;
        preamble << std::hexfloat << std::setprecision(std::numeric_limits<double>::digits10 + 1);
        preamble << "#define TYPE_MAX " << std::numeric_limits<double>::max() << std::endl;
        preamble << "#define TYPE_MIN " << std::numeric_limits<double>::min() << std::endl;

        srcs.push_back(preamble.str() + _sources);

        //std::cerr << "--------- preamble: ---------" << std::endl;
        //std::cerr << preamble.str() << std::endl << std::endl;
        //std::cerr << "Building program with the following sources " << std::endl;
        //std::cerr << preamble.str() + _sources << std::endl;
        //std::cerr << std::endl;
        //exit(1);

        cl_int err;
        cl::Program p{_runtime.getContext(), srcs, &err};
        if (err != CL_SUCCESS) {
            std::cerr << "Error in program creation <<" << std::endl;
            std::cerr << cl::errors::to_string(err);
            exit(1);
        }
        p.build();

        for (auto &[_, info] : p.getBuildInfo<CL_PROGRAM_BUILD_LOG>()) {
            //std::cerr << info << std::endl;
        }

        return p;
    }

    MinMax::ReduceKernel MinMax::getReduceKernel(unsigned int thread, bool isPow2) const {
        cl::Program p = getProgram(thread, isPow2);

        cl_int err;
        auto k = ReduceKernel(p, CL_KERNEL_NAME, &err);

        if (err != CL_SUCCESS) {
            std::cerr << "Error in program creation" << CL_SOURCE_FILE << std::endl;
            std::cerr << cl::errors::to_string(err);
            exit(1);
        }

        return k;
    }

    MinMax::ReduceKernel MinMax::getReduceSubKernel(unsigned int thread, bool isPow2) const {
        cl::Program p = getProgram(thread, isPow2);

        cl_int err;
        auto k = ReduceKernel(p, CL_KERNEL_SUB_NAME, &err);

        if (err != CL_SUCCESS) {
            std::cerr << "Error in kernel creation" << CL_KERNEL_NAME << std::endl;
            std::cerr << cl::errors::to_string(err);
            exit(1);
        }

        return k;
    }
}
