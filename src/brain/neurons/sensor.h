/*
 *   binary neurons network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef BNN_NEURONS_SENSOR_H
#define BNN_NEURONS_SENSOR_H

#include <vector>

#include "neuron.h"

namespace bnn
{

struct sensor final : neuron
{
    _word world_input_address;
    sensor(std::vector<bool>& world_input, _word world_input_address);
    void solve(brain &brn);
};

} // namespace bnn

#endif // BNN_NEURONS_SENSOR_H
