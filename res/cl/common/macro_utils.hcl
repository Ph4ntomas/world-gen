#ifndef COMMOM_MACRO_UTILS_HCL
#define COMMON_MACRO_UTILS_HCL

#define M_CONCAT(A, B) M_CONCAT_(A, B) //enable macro expansion with indirection
#define M_CONCAT_(A, B) A##B

#endif
