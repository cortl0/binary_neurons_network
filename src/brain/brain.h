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

#include <memory>
#include <thread>
#include <vector>

#include "config.hpp"
#include "state.hpp"
#include "random/random.h"
#include "simple_math.hpp"
#include "storage.hpp"

#include "neurons/neuron.h"
#include "neurons/binary.h"
#include "neurons/sensor.h"
#include "neurons/motor.h"

namespace bnn
{

class thread;

struct brain
{
public:
    u_word candidate_for_create_j;
    u_word candidate_for_create_k;
    u_word candidate_for_kill = 0;
    u_word quantity_of_neurons_in_power_of_two;
    u_word quantity_of_neurons;
    u_word quantity_of_neurons_binary;
    u_word quantity_of_initialized_neurons_binary = 0;
    u_word threads_count;
    state state_ = state::stopped;
    random::config random_config;

protected:
    u_word quantity_of_neurons_sensor;
    u_word quantity_of_neurons_motor;

private:
    u_word iteration = 0;
    u_word random_array_length_in_power_of_two;

public:
    std::unique_ptr<random::random> random_;
    std::vector<storage> storage_;
    std::vector<bool> world_input;
    std::vector<bool> world_output;
    std::vector<bnn::thread> threads;

public:
    virtual ~brain();
    brain() = delete;
    explicit brain(u_word quantity_of_neurons_in_power_of_two,
                   u_word input_length,
                   u_word output_length,
                   u_word threads_count_in_power_of_two = 0);
    bool get_output(u_word offset) const;
    void set_input(u_word offset, bool value);
    void start();

protected:
    const u_word& get_iteration() const;
    void stop();

private:
    std::thread main_thread;

    static void function(brain* brn);
};

} // namespace bnn

#endif // BNN_BRAIN_H
