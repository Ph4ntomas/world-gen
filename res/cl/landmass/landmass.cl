#include "common/template/real.hcl"

#ifndef NB_SEED
    #error "NB_SEED must be defined"
#endif

enum SeedType {
    Negative = 0,
    Positive = 1
}

typedef struct {
    SeedType type;
    RType2 pos;
    double range;
    double offset;
} seed_t;

/**
** TODO: Add a way to offset when calculating distance
*/
kernel void landmass(RType threshold, global RType const *data_in, global RType *data_out, constant seed_t const * seeds) {
    size_t data_off = get_global_id(0) + get_global_id(1) * get_global_size(0);
    unsigned int i = NB_SEED;
    RType input = data_in[data_off];
    RType2 pos = {(RType)get_global_id(0), (RType)get_global_id(1)};

    while (i-- > 0) {
        RType dist = abs(distance(seeds[i].pos, pos));
        bool cond = (seeds[i].range - dist >= 0);
        RType dist_coef = 1 - (dist / seeds[i].range);
        RType att = ((RType)seeds[i].type + dist_coef * seeds[i].coef + seeds[i].offset) * cond; // if type == pos, 1 + dist_coef * coef + offset, if type == neg 0 + ... 
        att += (1 - seeds[i].type) * (!cond); //

        input *= att;
    }

    data_out[data_off] = (RType)(input - threshold >= 0);
}
