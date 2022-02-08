/*
 *   binary neurons network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef BNN_RANDOM_CONFIG_H
#define BNN_RANDOM_CONFIG_H

#include "../config.h"

namespace bnn::random
{

struct config
{
    _word get_offset = 0;
    _word get_offset_in_word = 0;
    _word put_offset_start;
    _word put_offset_end;
    _word put_offset;
    _word put_offset_in_word = 0;

#ifdef DEBUG
    unsigned long long int debug_count_get = 0;
    unsigned long long int debug_count_put = 0;
    long long int debug_sum_put = 0;
#endif
};

} // namespace bnn::random

#endif // BNN_RANDOM_CONFIG_H
