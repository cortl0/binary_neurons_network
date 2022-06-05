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

#include "../../common/headers/config.hpp"
#include "random.h"

namespace bnn
{

class brain;

class m_sequence;

namespace random
{

class random;

} // namespace random

class thread final
{
public:
    u_word iteration = 0;
    u_word quantity_of_initialized_neurons_binary = 0;
    random::random::config random_config;
    bool in_work = false;

    thread();
    thread(brain*,
           u_word thread_number,
           u_word start_neuron,
           u_word length_in_us_in_power_of_two,
           random::random::config&);
    void start();

private:
    u_word length_in_us_in_power_of_two;
    u_word start_neuron;
    u_word thread_number;

#ifdef DEBUG
public:
    unsigned long long int debug_created = 0;
    unsigned long long int debug_killed = 0;
    u_word debug_average_consensus = 0;
    s_word debug_max_consensus = 0;
    u_word debug_max_consensus_binary_num = 0;
    u_word debug_max_consensus_motor_num = 0;
#endif

    char save_load_size;

public:
    brain* brain_;

private:
    void function();
};

} // namespace bnn

#endif // BNN_THREAD_H
