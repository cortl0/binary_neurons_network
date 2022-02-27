/*
 *   binary neurons network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef BNN_RANDOM_CONFIG_HPP
#define BNN_RANDOM_CONFIG_HPP

#include "../config.hpp"

namespace bnn::random
{

struct config
{
    u_word get_offset = 0;
    u_word get_offset_in_word = 0;
    u_word put_offset_start;
    u_word put_offset_end;
    u_word put_offset;
    u_word put_offset_in_word = 0;

#ifdef DEBUG
    unsigned long long int debug_count_get = 0;
    unsigned long long int debug_count_put = 0;
    long long int debug_sum_put = 0;
#endif
};

} // namespace bnn::random

#endif // BNN_RANDOM_CONFIG_HPP
