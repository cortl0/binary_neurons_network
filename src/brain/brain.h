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

struct brain
{
    _word candidate_for_kill = 0;
    _word quantity_of_neurons_in_power_of_two;
    _word quantity_of_neurons;
    _word quantity_of_neurons_binary;
    _word quantity_of_initialized_neurons_binary = 0;
    _word threads_count;

    state state_ = state::stopped;

    std::unique_ptr<random::random> random_;
    std::vector<storage> storage_;
    std::vector<bool> world_input;
    std::vector<bool> world_output;
    std::vector<bnn::thread> threads;

public:
    virtual ~brain();
    brain() = delete;
    explicit brain(_word random_array_length_in_power_of_two,
                   _word quantity_of_neurons_in_power_of_two,
                   _word input_length,
                   _word output_length,
                   _word threads_count_in_power_of_two = 0);
    const _word &get_input_length() const;
    bool get_output(_word offset) const;
    const _word &get_output_length() const;
    void set_input(_word offset, bool value);
    void start();

protected:
    _word quantity_of_neurons_sensor;
    _word quantity_of_neurons_motor;

    const _word &get_iteration() const;
    void stop();

private:
    _word iteration = 0;
    _word random_array_length_in_power_of_two;
    std::thread main_thread;

    static void function(brain* brn);
    void primary_filling();
};

} // namespace bnn

#endif // BNN_BRAIN_H
