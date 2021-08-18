/*
 *   binary neurons network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#include "../brain.h"

namespace bnn
{

brain::union_storage::neuron::neuron()
{
    neuron_type_ = neuron_type_neuron;
}

void brain::union_storage::neuron::solve(brain &brn, _word me, _word thread_number)
{
#ifdef DEBUG
    calculation_count++;
#endif

    switch (brn.us[me].neuron_.get_type())
    {
    case brain::union_storage::neuron::neuron_type_binary:
        brn.us[me].binary_.solve(brn, thread_number);
        break;
    case union_storage::neuron::neuron_type_sensor:
        brn.us[me].sensor_.solve(brn);
        break;
    case union_storage::neuron::neuron_type_motor:
        brn.us[me].motor_.solve(brn, me, thread_number);
        break;
    default:
        break;
    }
}

} // namespace bnn
