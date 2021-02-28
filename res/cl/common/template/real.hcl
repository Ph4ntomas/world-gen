#ifndef COMMON_TEMPLATE_REAL_HCL
#define COMMON_TEMPLATE_REAL_HCL

#include "common/macro_utils.hcl"

#ifndef RType
    #warning "Real Type not defined.\nDefaulting to float."
    #define RType float
#endif

#define RType2 M_CONCAT(RType, 2)
#define RType3 M_CONCAT(RType, 3)
#define RType4 M_CONCAT(RType, 4)
#define RType8 M_CONCAT(RType, 8)
#define RType16 M_CONCAT(RType, 16)

#endif
