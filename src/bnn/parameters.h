/*
 *   Binary Neurons Network
 *   created by Ilya Shishkin
 *   cortl@yandex.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef BNN_PARAMETERS_H
#define BNN_PARAMETERS_H

#include "random.h"
#include "state.h"

struct bnn_parameters
{
    u_word iteration{0};
    u_word quantity_of_initialized_neurons_binary{0};
    u_word candidate_for_kill{~u_word{0}};
    u_word size{0};
    bnn_random_config random_config;
    bnn_state state{bnn_state::stopped};
    bool training{false};
    bnn_error_codes bnn_error_code{bnn_error_codes::ok};
};

#endif // BNN_PARAMETERS_H
