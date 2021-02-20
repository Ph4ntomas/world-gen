#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>

#include "opencl/errors.hpp"
#include "algo/Reducer.hpp"

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
    Reducer::Reducer(std::string const &op, double init, opencl::IRuntime &rt): _runtime(rt) {
        std::ostringstream preamble;

        preamble << "#define REDUCE_OP " << op << std::endl;
        preamble << "#define T double" << std::endl;
        preamble << "#define INIT_VAL " << init << std::endl;

        _preamble = preamble.str();

        std::ifstream src(CL_SOURCE_FILE);
        _sources = std::string(std::istreambuf_iterator<char>(src), std::istreambuf_iterator<char>());
    }

    template <>
    Reducer::Reducer([[maybe_unused]] reduce_op::min_t, opencl::IRuntime &rt): Reducer("min", std::numeric_limits<double>::max(), rt) {}

    template <>
    Reducer::Reducer([[maybe_unused]] reduce_op::max_t, opencl::IRuntime &rt): Reducer("max", std::numeric_limits<double>::min(), rt) {}

    template <>
    Reducer::Reducer([[maybe_unused]] reduce_op::sum_t, opencl::IRuntime &rt): Reducer("sum", 0, rt) {}

    void Reducer::getNumThreadAndBlock(unsigned int size, unsigned int &thread, unsigned int &block) const {
        thread = (size < MAX_THREADS) ? nextPow2((size + 1) / 2) : MAX_THREADS;
        block = (size + thread * 2 - 1) / (thread * 2);
        block = std::min(MAX_BLOCK, block);
    }

    double Reducer::reduce(std::vector<double> const &data) {
        auto & ctx = _runtime.getContext();
        auto & queue = _runtime.getCommandQueue();

        unsigned int nThread = 0;
        unsigned int nBlock = 0;

        getNumThreadAndBlock(data.size(), nThread, nBlock);

        cl::Buffer in(ctx, CL_MEM_READ_ONLY, sizeof(double) * data.size());
        cl::Buffer out(ctx, CL_MEM_READ_WRITE, sizeof(double) * nBlock);

        queue.enqueueWriteBuffer(in, CL_FALSE, 0, sizeof(double) * data.size(), data.data());

        auto k = getReduceKernel(nThread, isPow2(data.size()));
        k.getKernel().setArg(3, sizeof(double) * nThread, NULL);
        k(cl::EnqueueArgs(queue, cl::NDRange(nBlock * nThread), cl::NDRange(nThread)), in, out, data.size());

        unsigned int block = 0;
        unsigned int thread = 0;

        for (unsigned s = nBlock; s > 1; s = (s + thread * 2 - 1) / (thread * 2)) {
            std::cerr << s << std::endl;
            getNumThreadAndBlock(s, thread, block);
            auto k = getReduceKernel(thread, isPow2(s));
            k.getKernel().setArg(3, sizeof(double) * thread, NULL);
            k(cl::EnqueueArgs(queue, cl::NDRange(block * thread), cl::NDRange(thread)),
                out, out, s
            );
        }

        double ret;
        queue.enqueueReadBuffer(out, CL_TRUE, 0, sizeof(double), &ret);
        return ret;
    }

    Reducer::ReduceKernel Reducer::getReduceKernel(unsigned int thread, bool isPow2) {
        cl::Program::Sources srcs;
        std::ostringstream preamble;

        preamble << _preamble;
        preamble << "#define IS_SZ_POW2 " << isPow2 << std::endl;
        preamble << "#define BLOCK_SIZE " << thread << std::endl;

        srcs.push_back(preamble.str() + _sources);

        //std::cerr << "--------- preamble: ---------" << std::endl;
        //std::cerr << preamble.str() << std::endl << std::endl;
        //std::cerr << "Building program with the following sources " << std::endl;
        //std::cerr << preamble.str() + _sources << std::endl;
        //std::cerr << std::endl;

        cl_int err;
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

        auto k = ReduceKernel(p, CL_KERNEL_NAME, &err);

        if (err != CL_SUCCESS) {
            std::cerr << "Error in kernel creation" << CL_KERNEL_NAME << std::endl;
            std::cerr << cl::errors::to_string(err);
            exit(1);
        }

        return k;
    }
}
