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

#include "../config.hpp"

namespace bnn
{

struct brain;

union storage;

namespace neurons
{

struct neuron
{
    enum class type : s_word
    {
        neuron,
        sensor,
        binary,
        motor
    };

    type type_;
    u_word level = 1;
    u_word life_counter = 0;
    bool output_new;
    bool output_old;
    char char_reserve_neuron[2];
    neuron();
    const type& get_type() const;
    void solve(brain&, u_word me, u_word thread_number);

#ifdef DEBUG
    u_word calculation_count = 0;
#endif
};

} // namespace neurons

} // namespace bnn

#endif // BNN_NEURONS_NEURON_H
