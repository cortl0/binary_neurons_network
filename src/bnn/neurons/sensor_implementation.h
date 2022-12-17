/*
 *   binary neurons network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef BNN_NEURONS_SENSOR_IMPLEMENTATION_H
#define BNN_NEURONS_SENSOR_IMPLEMENTATION_H

#include "neuron_implementation.h"

auto bnn_sensor_set = [](
        bnn_sensor* me,
        bool* world_input,
        u_word world_input_address
        ) -> void
{
    bnn_neuron_set(
            &me->neuron_,
            bnn_neuron::type::sensor,
            world_input[world_input_address],
            world_input[world_input_address]
            );

    me->world_input_address = world_input_address;
};

auto bnn_sensor_calculate = [](
        bnn_sensor* me,
        bnn_world* input,
        bnn_random* random,
        bnn_random_config* random_config
        ) ->void
{
    bnn_neuron_calculate(&me->neuron_);
    me->neuron_.output_old = me->neuron_.output_new;
    me->neuron_.output_new = input->data[me->world_input_address];
    bnn_neuron_push_random(random, &me->neuron_, random_config);
};

#endif // BNN_NEURONS_SENSOR_IMPLEMENTATION_H
