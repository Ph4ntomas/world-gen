typedef struct {
    T vmin;
    T vmax;
} bound_t;

kernel void normalize(global T * input, global T * output, bound_t from, bound_t to) {
    T v = input[get_global_id(0)];
    output[get_global_id(0)] = to.vmin +    ((v - from.vmin) * (to.vmax - to.vmin)) / 
                                            (from.vmax - from.vmin);
}
