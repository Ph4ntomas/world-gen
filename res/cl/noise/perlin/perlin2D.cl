typedef struct {
    double weight;
    double freq;
    double amp;
} par_t;

double2 smooth(double2 p) {
    return p * p * p * (p * (p * 6.0 - 15.0) + 10.0);
}

double lerp(double v0, double v1, double t) {
    return (1. - t) * v0 + t * v1;
}

double _perlin2D(double2 p, constant int const * perm, constant double2 const *grad) {
    int2 p_int = convert_int2(floor(p));
    double2 p_dt = p - (convert_double2(p_int));
    double2 s_dt = smooth(p_dt);

    int2 x_coord = (int2)(p_int.x, p_int.x + 1) & 255;
    int2 y_coord = (int2)(p_int.y, p_int.y + 1) & 255;

    double2 x_dt = (double2)(p_dt.x, p_dt.x - 1);
    double2 y_dt = (double2)(p_dt.y, p_dt.y - 1);

    double2 c_grad[] = {
        grad[perm[perm[x_coord[0]] + y_coord[0]]],
        grad[perm[perm[x_coord[1]] + y_coord[0]]],
        grad[perm[perm[x_coord[0]] + y_coord[1]]],
        grad[perm[perm[x_coord[1]] + y_coord[1]]]
    };

    double2 v_corn[] = {
        {x_dt[0], y_dt[0]},
        {x_dt[1], y_dt[0]},
        {x_dt[0], y_dt[1]},
        {x_dt[1], y_dt[1]},
    };

    return lerp(
        lerp(dot(c_grad[0], v_corn[0]), dot(c_grad[1], v_corn[1]), s_dt[0]),
        lerp(dot(c_grad[2], v_corn[2]), dot(c_grad[3], v_corn[3]), s_dt[0]),
        s_dt[1]
    );
}

kernel void perlin2D(double2 scale, double2 offset, constant int const *perm, constant double2 const *grad, global double *out) {
    double2 p = ((double2)(get_global_id(0), get_global_id(1)) + offset) * scale;

    out[get_global_id(0) + get_global_id(1) * get_global_size(0)] = _perlin2D(p, perm, grad);
}

kernel void fracPerlin2D(par_t param, double2 scale, double2 offset, constant int const *perm, constant double2 const *grad, global double *out) {
    double res = 0;
    double weight = param.weight;
    double amp = param.amp;
    double freq = param.freq;

    double2 p = ((double2)(get_global_id(0), get_global_id(1)) + offset) * scale;

    /*for (int i = 0; i < param.octave; ++i) {*/
        /*weight += amp;*/
        /*res += _perlin2D(p * freq, perm, grad) * amp;*/
    res = _perlin2D(p * freq, perm, grad) * amp;

        /*amp *= param.persistence;*/
        /*freq *= param.lacunarity;*/
    /*}*/

    out[get_global_id(0) + get_global_id(1) * get_global_size(0)] += res / weight;
}
