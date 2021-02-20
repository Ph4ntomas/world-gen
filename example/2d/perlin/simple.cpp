#include <iostream>
#include <algorithm>
#include <execution>

#include "opencl/DefaultRuntime.hpp"
#include "noise/perlin/Perlin2D.hpp"
#include "../../utils/PgmImage.hpp"

struct args_t {
    args_t(): run(true),
        x(1024), y(1024),
        s_x(1.), s_y(1.),
        o_x(0.), o_y(0.),
        oct(1), pers(1.), lac(1.)
    {}

    bool run;
    size_t x;
    size_t y;
    double s_x;
    double s_y;
    double o_x;
    double o_y;
    unsigned oct;
    double pers;
    double lac;
};

void print_usage(std::string name) {
    std::cout << "Usage: " << name << "[OPTIONS] [SIZEX] [SIZEY] [SCALEX] [SCALEY] [OFFSETX] [OFFSETY] [oct] [persistence] [lacunarity]" << std::endl;
}

void parse_oct(args_t &args, std::vector<std::string> const &sav) {
    args.oct = std::stoul(sav[7]);
    args.pers = std::stod(sav[8]);
    args.lac = std::stod(sav[9]);
}

void parse_off(bool split, args_t & args, std::vector<std::string> const &sav) {
    args.o_x = std::stod(sav[5]);
    args.o_y = std::stod(sav[5 + split]);
}

void parse_scale(bool split, args_t & args, std::vector<std::string> const &sav) {
    args.s_x = std::stod(sav[3]);
    args.s_y = std::stod(sav[3 + split]);
}

void parse_size(bool split, args_t & args, std::vector<std::string> const &sav) {
    args.x = std::stoul(sav[1]);
    args.y = std::stoul(sav[1 + split]);
}

args_t getArgs(int ac, char **av) {
    args_t ret;
    std::vector<std::string> sav;

    for (int i = 0; i < ac; ++i) {
        std::string cur = av[i];
        if (cur == "-h" || cur == "--help") {
            print_usage(sav[0]);
            ret.run = false;
            return ret;
        }
        sav.push_back(cur);
    }

    switch (ac) {
        case 10:
            parse_oct(ret, sav);
        case 7:
        case 6:
            parse_off(ac > 6, ret, sav);
        case 5:
        case 4:
            parse_scale((ac > 4), ret, sav);
        case 3:
        case 2:
            parse_size(ac > 2, ret, sav);
            break;
        case 9:
        case 8:
            print_usage(sav[0]);
            ret.run = false;
            break;
        default:
            break;
    }

    return ret;
}

unsigned char noise_to_char(double n) {
    return static_cast<unsigned char>(((n + 1.) / 2.) * 255.);
}

void print_param(args_t arg) {
    std::cout << "Running with the following config :" << std::endl;
    std::cout << "\tDimension: " << arg.x << "x" << arg.y << std::endl;
    std::cout << "\tScaling: x: " << arg.s_x << ", y:" << arg.s_y << std::endl;
    std::cout << "\tOffset: x: " << arg.o_x << ", y:" << arg.o_y << std::endl;
    std::cout << "Perlin Params : " << std::endl;
    std::cout << "\tOctave: " << arg.oct << std::endl;
    std::cout << "\tPersistence: " << arg.pers << std::endl;
    std::cout << "\tLacunarity: " << arg.lac << std::endl;
}

int main(int ac, char **av)
{
    worldgen::opencl::DefaultRuntime rt;
    worldgen::noise::Perlin2D p(static_cast<unsigned long>(std::chrono::high_resolution_clock::now().time_since_epoch().count()), rt);

    auto args = getArgs(ac, av);

    if (args.run) {
        print_param(args);
        auto perlin = p.generate(
                {args.x, args.y},
                {1. / args.s_x, 1. / args.s_y},
                {args.o_x, args.o_y},
                {args.oct, args.pers, args.lac, false}
        );

        PgmImage img(args.x, args.y);
        std::transform(std::execution::par_unseq, perlin.begin(), perlin.end(), img.getData().begin(), noise_to_char);

        img.writeToFile("out.pgm");
    }

    return 0;
}
