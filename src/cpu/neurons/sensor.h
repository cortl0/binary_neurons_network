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

namespace bnn::neurons
{

struct sensor final : neuron
{
    u_word world_input_address;
    sensor(std::vector<bool>& world_input, u_word world_input_address);
    static void construct(sensor*, std::vector<bool>& world_input, u_word world_input_address);
    void solve(brain&, const u_word = -1, const u_word = -1) override;
};

} // namespace bnn::neurons

#endif // BNN_NEURONS_SENSOR_H
