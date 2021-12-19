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

namespace bnn
{

struct brain;

union storage;

struct binary final : neuron
{
    enum neuron_binary_type
    {
        neuron_binary_type_free = 0,
        neuron_binary_type_in_work = 1
    };

    neuron_binary_type neuron_binary_type_ = neuron_binary_type_free;
    _word first;  // input adress
    _word second; // input adress
    _word first_life_number; // input life number
    _word second_life_number; // input life number
    bool first_mem;
    bool second_mem;
    char char_reserve_binary[2]; // reserve
    binary();
    neuron_binary_type get_type_binary(){ return neuron_binary_type_; }
    void init(brain &brn, _word thread_number, _word j, _word k, std::vector<storage> &us);
    bool create(brain &brn, _word thread_number);
    void kill(brain &brn, _word thread_number);
    void solve_body(std::vector<storage> &us);
    void solve(brain &brn, _word thread_number);
};

} // namespace bnn

#endif // BNN_NEURONS_BINARY_H
