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

brain::union_storage::sensor::sensor(std::vector<bool>& world_input, _word world_input_address_)
{
    neuron_type_ = neuron_type_sensor;
    world_input_address = world_input_address_;
    out_new = world_input[world_input_address];
    out_old = out_new;
}

void brain::union_storage::sensor::solve(brain &brn)
{
    out_old = out_new;
    out_new = brn.world_input[world_input_address];
}

} // namespace bnn
