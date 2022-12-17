/*
 *   binary neurons network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef BNN_RANDOM_H
#define BNN_RANDOM_H

#include "config.h"

struct bnn_random
{
    u_word* data{nullptr};
    u_word size{0};
    u_word size_in_power_of_two{0};
};

struct bnn_random_config
{
    u_word get_offset{0};
    u_word get_offset_in_word{0};
    u_word put_offset_start{0};
    u_word put_offset_end{0};
    u_word put_offset{0};
    u_word put_offset_in_word{0};

#ifdef DEBUG
    unsigned long long int debug_count_get{0};
    unsigned long long int debug_count_put{0};
    long long int debug_sum_put{0};
#endif
};

#endif // BNN_RANDOM_H
