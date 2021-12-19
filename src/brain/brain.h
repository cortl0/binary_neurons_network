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

#include <thread>
#include <vector>

#include "config.h"
#include "state.h"
#include "random/random.h"
#include "simple_math.h"

#include "neurons/neuron.h"
#include "neurons/binary.h"
#include "neurons/sensor.h"
#include "neurons/motor.h"

namespace bnn
{

class thread;

struct brain final
{
#ifdef BRAIN_FRIEND_H
    friend struct brain_friend;
#endif

    _word quantity_of_neurons_in_power_of_two;
    _word quantity_of_neurons;
    _word quantity_of_neurons_binary;
    _word quantity_of_neurons_sensor;
    _word quantity_of_neurons_motor;
    _word quantity_of_initialized_neurons_binary = 0;
    _word random_array_length_in_power_of_two;
    _word iteration = 0;
    _word candidate_for_kill = 0;
    _word threads_count;
    void (*debug_out)(brain*, _word &old_iteration) = nullptr;
    
    state state_ = stopped;

    std::unique_ptr<random::random> random_;
    std::vector<storage> storage_;
    std::vector<bool> world_input;
    std::vector<bool> world_output;
    std::vector<thread> threads;
    std::thread main_thread;

    static void function(brain* brn);
    void primary_filling();
    void stop();

public:
    ~brain();
    brain() = delete;
    explicit brain(_word random_array_length_in_power_of_two,
                   _word quantity_of_neurons_in_power_of_two,
                   _word input_length,
                   _word output_length,
                   _word threads_count_in_power_of_two = 0);
    void start();
    bool get_out(_word offset);
    _word get_output_length();
    _word get_input_length();
    void set_in(_word offset, bool value);
    _word get_iteration();
};

} // namespace bnn

#endif // BNN_BRAIN_H
