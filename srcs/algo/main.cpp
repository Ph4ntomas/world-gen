#include <iostream>
#include <iomanip>
#include <random>
#include <chrono>

#include "opencl/DefaultRuntime.hpp"
#include "algo/Reducer.hpp"

namespace wo = worldgen::opencl;
namespace wa = worldgen::algo;

int main(void)
{
    wo::DefaultRuntime drt;
    wa::Reducer reduce{wa::reduce_op::sum, drt};

    std::vector<double> data(1024 * 1024, 0.);

    std::mt19937 g(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::uniform_real_distribution d_r{-1., 1.};

    double sum = 0;

    for (auto &v : data) {
        v = d_r(g);
        sum += v;
    }

    double sum_gpu = reduce.reduce(data);

    std::cout << "Result" << std::endl;
    //std::cout << //std::hexfloat;
    //std::setprecision(std::numeric_limits<double>::digits10 + 1);
    std::cout << "\tCPU = " << sum << std::endl;
    std::cout << "\tGPU = " << sum_gpu << std::endl;

    return 0;
}
