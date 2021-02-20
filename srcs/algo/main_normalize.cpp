#include <random>
#include <iostream>
#include <chrono>
#include <limits>
#include <iomanip>

#include "opencl/DefaultRuntime.hpp"
#include "algo/MinMax.hpp"
#include "algo/Normalizer.hpp"
#include "algo/Reducer.hpp"

namespace wo = worldgen::opencl;
namespace wa = worldgen::algo;

int main(void)
{
    wo::DefaultRuntime drt;
    wa::MinMax reducer{drt};
    wa::Normalizer norma{drt};

    std::vector<double> data(1024 * 1024, 0.);

    std::mt19937 g(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::uniform_real_distribution d_r{-.5, .5};

    double min_cpu = std::numeric_limits<double>::max();
    double max_cpu = std::numeric_limits<double>::min();

    for (auto &v : data) {
        v = d_r(g);
    }

    auto [min_gpu_pre, max_gpu_pre] = reducer.reduce(data);

    data = norma.normalize(data, -1., 1.);
    auto [min_gpu, max_gpu] = reducer.reduce(data);

    std::cout << std::setprecision(std::numeric_limits<double>::digits10 +1);
    std::cout << "Result" << std::endl;
    std::cout << "Pre:" << std::endl;
    std::cout << "\tmin = " << min_gpu_pre << std::endl;
    std::cout << "\tmax = " << max_gpu_pre << std::endl;
    std::cout << "Post:" << std::endl;
    std::cout << "\tmin = " << min_gpu << std::endl;
    std::cout << "\tmax = " << max_gpu << std::endl;

    return 0;
}
