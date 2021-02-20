#include "opencl/exceptions.hpp"

namespace cl::exceptions {

    std::string ClException::err_str(ErrorKind kind, cl::int1 err,
            const char * file, int line, char const *fun) {
        using namespace std::string_literals;
        std::string ret;

        ret += "OpenCL error in file "s;
        ret += ""s + fun + ":"s + std::to_string(line) + ", in function: "s;
        ret += fun + ":\n"s;
        ret += "\t"s + cl::errors::to_string(err);
        return ret;
    }

    ClException::ClException(ErrorKind k, cl::int1 e, const char * fi, int l, char const * fu):
        std::runtime_error(err_str(k, e, fi, l, fu)), _kind(k), _err(e)
    {}

    ClException::~ClException() {}

    ClException::ClException(ClException const &oth) noexcept
        : std::runtime_error(oth), _kind(oth._kind), _err(oth._err)
        {}

    ErrorKind ClException::getErrorKind() const noexcept {
        return _kind;
    }

    cl::int1 ClException::getError() const noexcept {
        return _err;
    }
}
