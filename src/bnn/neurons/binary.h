/*
 *   binary neurons network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef BNN_NEURONS_BINARY_H
#define BNN_NEURONS_BINARY_H

#include "neuron.h"

struct bnn_binary
{
//    struct input_neuron
//    {
//        u_word address;
//        u_word life_counter;
//        bool memory;
//    };

    bnn_neuron neuron_;
//    input_neuron first;
//    input_neuron second;
    u_word first_input_address;
    u_word second_input_address;
    u_word first_input_life_counter;
    u_word second_input_life_counter;
    bool first_input_memory;
    bool second_input_memory;
    bool in_work{false};
};

#endif // BNN_NEURONS_BINARY_H
