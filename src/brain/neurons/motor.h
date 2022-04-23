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

namespace bnn::neurons
{

struct motor final : neuron
{
    struct binary_neuron
    {
        u_word address;
        u_word life_counter;
        s_word consensus = 0;
        binary_neuron() = delete;
        explicit binary_neuron(u_word address, u_word life_counter, s_word consensus);
    };

    u_word world_output_address;
    s_word accumulator = 0;

#ifdef DEBUG
    u_word debug_average_consensus = 0;
#endif

    char save_load_size;

    std::map<u_word, binary_neuron> binary_neurons;

    motor(const std::vector<bool>& world_output, u_word world_output_address);
    void solve(brain&, const u_word thread_number, const u_word me) override;
};

} // namespace bnn::neurons

#endif // BNN_NEURONS_MOTOR_H
