/*#define T double //data type*/
/*#define T2 double2 //data type*/
/*#define BLOCK_SIZE 128 //number of thread in a block */
/*#define IS_SZ_POW2 false //enable optimization, only if user is sure that size is a power of 2*/
/*#define TYPE_MAX 0 //initial value */
/*#define TYPE_MIN 0 //Initial value*/

T2 minmax_fst(T2 lhs, T rhs) {
    lhs.x = lhs.x < rhs ? lhs.x : rhs;
    lhs.y = lhs.y > rhs ? lhs.y : rhs;
    return lhs;
}

T2 minmax(T2 lhs, T2 rhs) {
    lhs.x = lhs.x < rhs.x ? lhs.x : rhs.x;
    lhs.y = lhs.y > rhs.y ? lhs.y : rhs.y;
    return lhs;
}

/*
** Common function
*/
void _reduce(unsigned int tid, local volatile T2 * tmp) {
    barrier(CLK_LOCAL_MEM_FENCE); //waiting for other thread to finish
    if (BLOCK_SIZE >= 512) { 
        if (tid < 256) 
            tmp[tid] = minmax(tmp[tid], tmp[tid + 256]); 
        barrier(CLK_LOCAL_MEM_FENCE); 
    }
    if (BLOCK_SIZE >= 256) {
        if (tid < 128) 
            tmp[tid] = minmax(tmp[tid], tmp[tid + 128]); 
        barrier(CLK_LOCAL_MEM_FENCE); 
    }
    if (BLOCK_SIZE >= 128) {
        if (tid < 64) 
            tmp[tid] = minmax(tmp[tid], tmp[tid + 64]); 
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid < 32) {
        if (BLOCK_SIZE >= 64) tmp[tid] = minmax(tmp[tid], tmp[tid + 32]);
        if (BLOCK_SIZE >= 32) tmp[tid] = minmax(tmp[tid], tmp[tid + 16]);
        if (BLOCK_SIZE >= 16) tmp[tid] = minmax(tmp[tid], tmp[tid + 8]);
        if (BLOCK_SIZE >= 8) tmp[tid] = minmax(tmp[tid], tmp[tid + 4]);
        if (BLOCK_SIZE >= 4) tmp[tid] = minmax(tmp[tid], tmp[tid + 2]);
        if (BLOCK_SIZE >= 2) tmp[tid] = minmax(tmp[tid], tmp[tid + 1]);
    }

}

/*
** First reduction
*/
kernel void reduce_minmax(global T * input, global T2 * output, unsigned int size, local volatile T2 * tmp) {
    unsigned int tid = get_local_id(0);
    unsigned int i = get_group_id(0) * get_local_size(0) * 2 + tid; //skip previous groups' data
    unsigned skip = BLOCK_SIZE * 2 * get_num_groups(0); //each thread skip a certain number of element.
    tmp[tid].x = TYPE_MAX;
    tmp[tid].y = TYPE_MIN;

    while (i < size) {
        tmp[tid] = minmax_fst(tmp[tid],  input[i]);
        if (IS_SZ_POW2 || i + BLOCK_SIZE < size) // if IS_SZ_POW2 is set, this check gets optimized away
            tmp[tid] = minmax_fst(tmp[tid], input[i + BLOCK_SIZE]);
        i += skip;
    }

    _reduce(tid, tmp);

    if (tid == 0)
        output[get_group_id(0)] = tmp[0];
}

/*
** Subsequent reductions
*/
kernel void reduce_minmax_sub(global T2 * input, global T2 * output, unsigned int size, local volatile T2 * tmp) {
    unsigned int tid = get_local_id(0);
    unsigned int i = get_group_id(0) * get_local_size(0) * 2 + tid; //skip previous groups' data
    unsigned skip = BLOCK_SIZE * 2 * get_num_groups(0); //each thread skip a certain number of element.
    tmp[tid].x = TYPE_MAX;
    tmp[tid].y = TYPE_MIN;

    while (i < size) {
        tmp[tid] = minmax(tmp[tid], input[i]);
        if (IS_SZ_POW2 || i + BLOCK_SIZE < size) // if IS_SZ_POW2 is set, this check gets optimized away
            tmp[tid] = minmax(tmp[tid], input[i + BLOCK_SIZE]);
        i += skip;
    }

    _reduce(tid, tmp);

    if (tid == 0)
        output[get_group_id(0)] = tmp[0];
}
