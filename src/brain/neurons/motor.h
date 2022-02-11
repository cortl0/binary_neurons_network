/*
 *   binary neurons network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef BNN_NEURONS_MOTOR_H
#define BNN_NEURONS_MOTOR_H

#include <map>
#include <vector>

#include "neuron.h"

namespace bnn
{

struct motor final : neuron
{
    struct binary_neuron
    {
        _word adress; // binary neuron adress
        _word life_number;
        int consensus = 0;
        binary_neuron() = delete;
        explicit binary_neuron(_word adress, _word life_number, int consensus);
    };

    _word world_output_address;
    int accumulator = 0;
    std::map<_word, binary_neuron>* binary_neurons;

    motor(std::vector<bool>& world_output, _word world_output_address_);
    void solve(brain &brn, const _word &me, const _word &thread_number);

#ifdef DEBUG
    _word debug_average_consensus = 0;
#endif
};

} // namespace bnn

#endif // BNN_NEURONS_MOTOR_H
