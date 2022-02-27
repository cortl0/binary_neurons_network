/*
 *   binary neurons network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#include "neuron.h"
#include "../brain.h"
#include "../storage.hpp"

namespace bnn::neurons
{

neuron::neuron()
{
    type_ = neuron::type::neuron;
}

const neuron::type& neuron::get_type() const
{
    return type_;
}

void neuron::solve(brain& b, const u_word me, const u_word thread_number)
{
#ifdef DEBUG
    calculation_count++;
#endif

    switch (b.storage_[me].neuron_.get_type())
    {
    case type::binary:
        b.storage_[me].binary_.solve(b, thread_number);
        break;
    case type::sensor:
        b.storage_[me].sensor_.solve(b);
        break;
    case type::motor:
        b.storage_[me].motor_.solve(b, me, thread_number);
        break;
    default:
        break;
    }
}

} // namespace bnn::neurons
