/*
 *   Binary Neurons Network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef BNN_THREAD_H
#define BNN_THREAD_H

#include <thread>

#include "config.h"
#include "state.h"
#include "random/config.h"

namespace bnn
{

struct brain;

class m_sequence;

namespace random
{
class random;
}

class thread final
{
public:
    _word iteration = 0;
    _word quantity_of_initialized_neurons_binary = 0;
    random::config random_config;
    state state_ = state::stopped;

    thread(brain*,
           _word thread_number,
           _word start_neuron,
           _word length_in_us_in_power_of_two,
           random::config &random_config);
    void start();

private:
    brain *brain_;
    _word length_in_us_in_power_of_two;
    _word start_neuron;
    std::thread thread_;
    _word thread_number;

    static void function(thread* thread_, brain* brn, _word start_in_us, _word length_in_us_in_power_of_two);

#ifdef DEBUG
public:
    unsigned long long int debug_created = 0;
    unsigned long long int debug_killed = 0;
    _word debug_average_consensus = 0;
    _word debug_max_consensus = 0;
    _word debug_max_consensus_binary_num = 0;
    _word debug_max_consensus_motor_num = 0;
#endif
};

} // namespace bnn

#endif // BNN_THREAD_H
