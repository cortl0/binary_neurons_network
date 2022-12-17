/*
 *   Binary Neurons Network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef BNN_BRAIN_H
#define BNN_BRAIN_H

#include "bnn/bnn.h"

namespace bnn
{

//class brain;

//class m_sequence;

//namespace random
//{

//class random;

//} // namespace random

class brain
{
public:
    bnn_bnn* bnn{nullptr};
//    u_word iteration = 0;
//    u_word quantity_of_initialized_neurons_binary = 0;
//    random::random::config random_config;
//    bool in_work = false;

//    thread();
    brain(
            u_word quantity_of_neurons_in_power_of_two,
            u_word input_length,
            u_word output_length,
            u_word threads_count_in_power_of_two = 0
            );
    void start();

    void stop();
    void set_input(u_word i, bool value);
    bool get_output(u_word i);

//private:
//    u_word length_in_us_in_power_of_two;
//    u_word start_neuron;
//    u_word thread_number;

//#ifdef DEBUG
//public:
//    unsigned long long int debug_created = 0;
//    unsigned long long int debug_killed = 0;
//    u_word debug_average_consensus = 0;
//    s_word debug_max_consensus = 0;
//    u_word debug_max_consensus_binary_num = 0;
//    u_word debug_max_consensus_motor_num = 0;
//#endif

//    char save_load_size;

//public:
//    brain* brain_;

//private:
//    void function();
};

} // namespace bnn

#endif // BNN_BRAIN_H
