/*#define T double //data type*/
/*#define BLOCK_SIZE 128 //number of thread in a block */
/*#define REDUCE_OP sum // Reduction Operator*/
/*#define INIT_VAL 0 //Initial value*/
/*#define IS_SZ_POW2 false //enable optimization, only if user is sure that size is a power of 2*/

T sum(T lhs, T rhs) {
    return lhs + rhs;
}

kernel void reduce(global T * input, global T * output, unsigned int size, local volatile T * tmp) {
    unsigned int tid = get_local_id(0);
    unsigned int i = get_group_id(0) * get_local_size(0) * 2 + tid; //skip previous groups' data
    unsigned skip = BLOCK_SIZE * 2 * get_num_groups(0); //each thread skip a certain number of element.
    tmp[tid] = INIT_VAL;
    printf("%f\n", tmp[tid]);

    while (i < size) {
        tmp[tid] = REDUCE_OP(tmp[tid], input[i]);
        if (IS_SZ_POW2 || i + BLOCK_SIZE < size) // if IS_SZ_POW2 is set, this check gets optimized away
            tmp[tid] = REDUCE_OP(tmp[tid], input[i + BLOCK_SIZE]);
        i += skip;
    }

    barrier(CLK_LOCAL_MEM_FENCE); //waiting for other thread to finish
    if (BLOCK_SIZE >= 512) { 
        if (tid < 256) 
            tmp[tid] = REDUCE_OP(tmp[tid], tmp[tid + 256]); 
        barrier(CLK_LOCAL_MEM_FENCE); 
    }
    if (BLOCK_SIZE >= 256) {
        if (tid < 128) 
            tmp[tid] = REDUCE_OP(tmp[tid], tmp[tid + 128]); 
        barrier(CLK_LOCAL_MEM_FENCE); 
    }
    if (BLOCK_SIZE >= 128) {
        if (tid < 64) 
            tmp[tid] = REDUCE_OP(tmp[tid], tmp[tid + 64]); 
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid < 32) {
        if (BLOCK_SIZE >= 64) tmp[tid] = REDUCE_OP(tmp[tid], tmp[tid + 32]);
        if (BLOCK_SIZE >= 32) tmp[tid] = REDUCE_OP(tmp[tid], tmp[tid + 16]);
        if (BLOCK_SIZE >= 16) tmp[tid] = REDUCE_OP(tmp[tid], tmp[tid + 8]);
        if (BLOCK_SIZE >= 8) tmp[tid] = REDUCE_OP(tmp[tid], tmp[tid + 4]);
        if (BLOCK_SIZE >= 4) tmp[tid] = REDUCE_OP(tmp[tid], tmp[tid + 2]);
        if (BLOCK_SIZE >= 2) tmp[tid] = REDUCE_OP(tmp[tid], tmp[tid + 1]);
    }

    if (tid == 0)
        output[get_group_id(0)] = tmp[0];
}
