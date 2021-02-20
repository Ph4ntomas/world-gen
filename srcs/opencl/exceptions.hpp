#ifndef opencl_exceptions_hpp__
#define opencl_exceptions_hpp__

#include <exception>
#include <stdexcept>

#include "opencl/cl.hpp"
#include "opencl/errors.hpp"

#define cl_throwIfError(err) cl::exceptions::_throwIfError((err), __FILE__, __LINE__, __func__)
#define cl_throwIfErrorKind(kind, err) cl::exceptions::_throwIfError<(kind)>((err), __FILE__, __LINE__, __func__)

namespace cl::exceptions {
    enum struct ErrorKind {
        Program,
        Kernel,
        Buffer,
        Other,
        Unspecified
    };

    class ClException : public std::runtime_error {
        static std::string err_str(ErrorKind kind, cl::int1 err, const char * file, int line, char const *fun);
        public:
            ClException(ErrorKind kind, cl::int1 err, const char * file, int line, char const * fun);
            ClException(ClException const &oth) noexcept;
            virtual ~ClException();

            [[nodiscard]]
            virtual ErrorKind getErrorKind() const noexcept final;
            [[nodiscard]]
            virtual cl::int1 getError() const noexcept final;

        private:
            ErrorKind _kind;
            cl::int1 _err;
    };

    template <ErrorKind kind>
    class GenericClException : public ClException {
        public:
            GenericClException(cl::int1 err, char const * file, int line, char const *fun)
                : ClException(kind, err, file, line, fun) {}
            GenericClException(GenericClException const &oth) noexcept
                : ClException(oth) {}
            virtual ~GenericClException() {}

        private:
    };

    template <ErrorKind k = ErrorKind::Unspecified>
    void _throwIfError(cl::int1 err, char const *file, int line, char const *fun) {
        if (err != CL_SUCCESS)
            throw GenericClException<k>(err, file, line, fun);
    }
}


#endif /* end of include guard: opencl_exceptions_hpp__ */
