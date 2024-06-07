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

auto bnn_sensor_set = [BNN_LAMBDA_REFERENCE](
        bnn_sensor* me,
        bool output_new,
        bool output_old,
        u_word world_input_address
        ) -> void
{
    bnn_neuron_set(
            &me->neuron_,
            bnn_neuron::type::sensor,
            output_new,
            output_old
            );

    me->world_input_address = world_input_address;
};

auto bnn_sensor_calculate = [BNN_LAMBDA_REFERENCE](
        bnn_bnn* bnn,
        bnn_sensor* me,
        bnn_random_config* random_config
        ) ->void
{
    me->neuron_.output_old = me->neuron_.output_new;
    me->neuron_.output_new = bnn->input_.data[me->world_input_address];
    bnn_neuron_push_random(&bnn->random_, &me->neuron_, random_config);
};

#endif // BNN_NEURONS_SENSOR_IMPLEMENTATION_H
