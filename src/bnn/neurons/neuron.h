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

#include "../../common/headers/config.hpp"

namespace bnn
{

class brain;

union storage;

namespace neurons
{

struct neuron
{
    enum class type : u_word
    {
        neuron,
        sensor,
        binary,
        motor
    };

    type type_ = type::neuron;
    u_word level = 1;
    u_word life_counter = 0;
    bool output_new;
    bool output_old;
    neuron();
    const type& get_type() const;
    void put_random(brain& b, const u_word thread_number);
    virtual void solve(brain&, const u_word thread_number, const u_word me);

#ifdef DEBUG
    u_word calculation_count = 0;
#endif
};

} // namespace neurons

} // namespace bnn

#endif // BNN_NEURONS_NEURON_H
