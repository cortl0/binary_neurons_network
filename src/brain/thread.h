/*
 *   binary neurons network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef THREAD_H
#define THREAD_H

#include <thread>

#include "config.h"

namespace bnn
{

struct brain;
class m_sequence;
class random_put_get;

class thread
{
private:
    _word thread_number;
    _word length_in_us_in_power_of_two;
public:
    _word start_neuron;
private:
public:
    _word quantity_of_initialized_neurons_binary = 0;
    _word random_array_length_in_power_of_two;
    std::thread thread_;
    _word iteration = 0;
    bool in_work = false;
#ifdef DEBUG
    unsigned long long int debug_created = 0;
    unsigned long long int debug_killed = 0;
    _word debug_average_consensus = 0;
    _word debug_max_consensus = 0;
    _word debug_max_consensus_binary_num = 0;
    _word debug_max_consensus_motor_num = 0;
#endif
    std::unique_ptr<random_put_get> rndm;
    thread(brain* brn,
           _word thread_number,
           _word start_neuron,
           _word length_in_us_in_power_of_two,
           _word random_array_length_in_power_of_two,
           m_sequence& m_sequence);
    static void function(brain* brn, _word thread_number, _word start_in_us, _word length_in_us_in_power_of_two);
};

} // namespace bnn

#endif // THREAD_H
