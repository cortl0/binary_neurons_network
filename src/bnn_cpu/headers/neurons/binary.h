/*
 *   binary neurons network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef BNN_NEURONS_BINARY_H
#define BNN_NEURONS_BINARY_H

#include <vector>

#include "neuron.h"

namespace bnn::neurons
{

struct binary final : neuron
{
    u_word first_input_address;
    u_word second_input_address;
    u_word first_input_life_counter;
    u_word second_input_life_counter;
    bool first_input_memory;
    bool second_input_memory;
    bool in_work = false;
    binary();
    void init(brain&, u_word thread_number, u_word first, u_word second, const std::vector<std::shared_ptr<neuron>>&);
    void create(brain&, u_word thread_number, const u_word me);
    void kill(brain&, u_word thread_number);
    void solve_body(const std::vector<std::shared_ptr<neuron>>&);
    void solve(brain&, const u_word thread_number, const u_word me) override;
};

} // namespace bnn::neurons

#endif // BNN_NEURONS_BINARY_H
