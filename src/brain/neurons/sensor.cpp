/*
 *   binary neurons network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#include "sensor.h"
#include "../brain.h"

namespace bnn::neurons
{

sensor::sensor(std::vector<bool>& world_input, u_word world_input_address)
    : world_input_address(world_input_address)
{
    type_ = type::sensor;
    output_new = world_input[world_input_address];
    output_old = output_new;
}

void sensor::solve(brain& b)
{
    output_old = output_new;
    output_new = b.world_input[world_input_address];
}

} // namespace bnn::neurons
