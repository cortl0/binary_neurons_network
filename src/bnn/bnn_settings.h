/*
 *   Binary Neurons Network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef BNN_SETTINGS_H
#define BNN_SETTINGS_H

#include "types.h"

struct bnn_settings
{
    u_word quantity_of_neurons_in_power_of_two;
    u_word input_length;
    u_word output_length;
    u_word threads_count_in_power_of_two;
};

#endif // BNN_SETTINGS_H
