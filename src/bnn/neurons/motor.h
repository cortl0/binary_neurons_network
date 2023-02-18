/*
 *   binary neurons network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef BNN_NEURONS_MOTOR_H
#define BNN_NEURONS_MOTOR_H

#include "neuron.h"

struct bnn_motor
{
    struct binary
    {
        u_word address{~u_word{0}};
        u_word life_counter{0};
        s_word consensus{0};
        bool present{false};
    };

    struct binaries
    {
        bnn_motor::binary* data{nullptr};
        u_word size{0};
        u_word size_per_motor{0};
    };

    bnn_neuron neuron_;
    u_word world_output_address{~u_word{0}};
    s_word accumulator{0};
    u_word binaries_offset{~u_word{0}};
    u_word binaries_filled_size{0};

#ifdef DEBUG
    u_word debug_max_consensus{0};
    u_word debug_average_consensus{0};
#endif
};

#endif // BNN_NEURONS_MOTOR_H
