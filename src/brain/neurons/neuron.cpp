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
#include "../storage.h"

namespace bnn
{

neuron::neuron()
{
    type_ = neuron::type::neuron;
}

const neuron::type &neuron::get_type() const
{
    return type_;
}

void neuron::solve(brain &brn, _word me, _word thread_number)
{
#ifdef DEBUG
    calculation_count++;
#endif

    switch (brn.storage_[me].neuron_.get_type())
    {
    case type::binary:
        brn.storage_[me].binary_.solve(brn, thread_number);
        break;
    case type::sensor:
        brn.storage_[me].sensor_.solve(brn);
        break;
    case type::motor:
        brn.storage_[me].motor_.solve(brn, me, thread_number);
        break;
    default:
        break;
    }
}

} // namespace bnn
