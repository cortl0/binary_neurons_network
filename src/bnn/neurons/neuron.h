/*
 *   binary neurons network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef BNN_NEURONS_NEURON_H
#define BNN_NEURONS_NEURON_H

#include "bnn/config.h"

struct bnn_neuron
{
    enum type
    {
        neuron,
        sensor,
        binary,
        motor
    };

    type type_{type::neuron};
    u_word level{1};
    u_word life_counter{0};
    bool output_new{true};
    bool output_old{false};

#ifdef DEBUG
    u_word calculation_count{0};
#endif
};

#endif // BNN_NEURONS_NEURON_H
