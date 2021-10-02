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
    neuron_type_ = neuron_type_neuron;
}

void neuron::solve(brain &brn, _word me, _word thread_number)
{
#ifdef DEBUG
    calculation_count++;
#endif

    switch (brn.storage_[me].neuron_.get_type())
    {
    case neuron::neuron_type_binary:
        brn.storage_[me].binary_.solve(brn, thread_number);
        break;
    case neuron::neuron_type_sensor:
        brn.storage_[me].sensor_.solve(brn);
        break;
    case neuron::neuron_type_motor:
        brn.storage_[me].motor_.solve(brn, me, thread_number);
        break;
    default:
        break;
    }
}

} // namespace bnn
