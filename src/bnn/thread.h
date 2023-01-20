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

#include "random.h"

struct bnn_thread
{
    //u_word big_iteration{0};
    u_word iteration{0};
    u_word quantity_of_initialized_neurons_binary{0};
    bnn_random_config random_config;
    u_word start_neuron{0};
    u_word length_in_us_in_power_of_two{0};
    u_word thread_number{0};
    bool in_work{false};

#ifdef DEBUG
    unsigned long long int debug_created{0};
    unsigned long long int debug_killed{0};
    u_word debug_average_consensus{0};
    s_word debug_max_consensus{0};
    u_word debug_max_consensus_binary_num{0};
    u_word debug_max_consensus_motor_num{0};
#endif

};

struct bnn_threads
{
    bnn_thread* data{nullptr};
    u_word size{0};
    u_word size_in_power_of_two{0};
    u_word neurons_per_thread{0};
};

#endif // BNN_THREAD_H
