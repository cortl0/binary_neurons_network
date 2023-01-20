/*
 *   binary neurons network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef BNN_NEURONS_NEURON_IMPLEMENTATION_H
#define BNN_NEURONS_NEURON_IMPLEMENTATION_H

#include "bnn/bnn.h"
#include "bnn/random_implementation.h"

auto bnn_neuron_set = [BNN_LAMBDA_REFERENCE](
        bnn_neuron* me,
        bnn_neuron::type type,
        bool output_new = true,
        bool output_old = false
        ) -> void
{
    me->type_ = type;
    me->level = u_word{1};
    me->life_counter = u_word{0};
    me->output_new = output_new;
    me->output_old = output_old;

#ifdef DEBUG
    me->calculation_count = u_word{0};
#endif
};

auto bnn_neuron_push_random = [BNN_LAMBDA_REFERENCE](
        bnn_random* random,
        bnn_neuron* me,
        bnn_random_config* random_config
        ) -> void
{
    if(me->output_new != me->output_old)
        bnn_random_push(random, me->output_new, random_config);
};

auto bnn_neuron_calculate = [BNN_LAMBDA_REFERENCE](
        bnn_neuron* me
        ) -> void
{
#ifdef DEBUG
    me->calculation_count++;
#endif
};

#endif // BNN_NEURONS_NEURON_IMPLEMENTATION_H
