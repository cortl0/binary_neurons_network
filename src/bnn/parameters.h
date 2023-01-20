/*
 *   Binary Neurons Network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef BNN_PARAMETERS_H
#define BNN_PARAMETERS_H

#include "random.h"

struct bnn_parameters
{
    u_word iteration{0};
    u_word quantity_of_initialized_neurons_binary{0};
    u_word candidate_for_kill{~u_word{0}};
    bnn_random_config random_config;
    bool start{false};
    bool stop{false};
};

#endif // BNN_PARAMETERS_H
