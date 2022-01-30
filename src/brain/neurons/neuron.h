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

#include "../config.h"

namespace bnn
{

struct brain;

struct neuron
{
    enum neuron_type
    {
        neuron_type_neuron = 0,
        neuron_type_sensor = 1,
        neuron_type_binary = 2,
        neuron_type_motor = 3
    };
    neuron_type neuron_type_;
    _word level = 1;
    _word life_number = 0;
    bool out_new;
    bool out_old;
    char char_reserve_neuron[2]; // reserve
    neuron();
    const neuron_type &get_type() const;
    void solve(brain &brn, _word me, _word thread_number);

#ifdef DEBUG
    _word calculation_count = 0;
#endif
};

} // namespace bnn

#endif // BNN_NEURONS_NEURON_H
