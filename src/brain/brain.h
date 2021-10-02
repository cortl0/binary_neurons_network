/*
 *   binary neurons network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef BRAIN_H
#define BRAIN_H

#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <thread>
#include <vector>
#include <unistd.h>

#include "config.h"
#include "state.h"
#include "random_put_get.h"
#include "simple_math.h"

#include "neurons/neuron.h"
#include "neurons/binary.h"
#include "neurons/sensor.h"
#include "neurons/motor.h"

namespace bnn
{

class thread;

struct brain
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
    
    state state_ = stopped;

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
    brain(_word random_array_length_in_power_of_two,
          _word quantity_of_neurons_in_power_of_two,
          _word input_length,
          _word output_length,
          _word threads_count = 1);
    void start();
    bool get_out(_word offset);
    _word get_output_length();
    _word get_input_length();
    void set_in(_word offset, bool value);
    _word get_iteration();
};

} // namespace bnn

#endif // BRAIN_H
