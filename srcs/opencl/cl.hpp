#ifndef _pglib_opencl_cl_hpp_
#define _pglib_opencl_cl_hpp_

#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <CL/opencl.hpp>

namespace cl {
    using char1 = cl_char;
    using char2 = cl_char2;
    using char3 = cl_char3;
    using char4 = cl_char4;
    using char8 = cl_char8;
    using char16 = cl_char16;

    using uchar1 = cl_uchar;
    using uchar2 = cl_uchar2;
    using uchar3 = cl_uchar3;
    using uchar4 = cl_uchar4;
    using uchar8 = cl_uchar8;
    using uchar16 = cl_uchar16;

    using short1 = cl_short;
    using short2 = cl_short2;
    using short3 = cl_short3;
    using short4 = cl_short4;
    using short8 = cl_short8;
    using short16 = cl_short16;

    using ushort1 = cl_ushort;
    using ushort2 = cl_ushort2;
    using ushort3 = cl_ushort3;
    using ushort4 = cl_ushort4;
    using ushort8 = cl_ushort8;
    using ushort16 = cl_ushort16;

    using int1 = cl_int;
    using int2 = cl_int2;
    using int3 = cl_int3;
    using int4 = cl_int4;
    using int8 = cl_int8;
    using int16 = cl_int16;

    using uint1 = cl_uint;
    using uint2 = cl_uint2;
    using uint3 = cl_uint3;
    using uint4 = cl_uint4;
    using uint8 = cl_uint8;
    using uint16 = cl_uint16;

    using long1 = cl_long;
    using long2 = cl_long2;
    using long3 = cl_long3;
    using long4 = cl_long4;
    using long8 = cl_long8;
    using long16 = cl_long16;

    using ulong1 = cl_ulong;
    using ulong2 = cl_ulong2;
    using ulong3 = cl_ulong3;
    using ulong4 = cl_ulong4;
    using ulong8 = cl_ulong8;
    using ulong16 = cl_ulong16;

    using float1 = cl_float;
    using float2 = cl_float2;
    using float3 = cl_float3;
    using float4 = cl_float4;
    using float8 = cl_float8;
    using float16 = cl_float16;

    using double1 = cl_double;
    using double2 = cl_double2;
    using double3 = cl_double3;
    using double4 = cl_double4;
    using double8 = cl_double8;
    using double16 = cl_double16;
}

#endif //_pglib_opencl_cl_hpp_
