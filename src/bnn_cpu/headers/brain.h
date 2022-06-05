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

#include "../../common/headers/config.hpp"
#include "neurons/binary.h"
#include "neurons/motor.h"
#include "neurons/sensor.h"
#include "random.h"
#include "neurons/storage.hpp"

namespace bnn
{

class thread;

class brain
{
public:
    u_word candidate_for_create_j;
    u_word candidate_for_create_k;
    u_word candidate_for_kill = -1;
    u_word quantity_of_neurons_in_power_of_two;
    u_word quantity_of_neurons;
    u_word quantity_of_neurons_binary;
    u_word quantity_of_initialized_neurons_binary = 0;
    bool to_work = false;
    bool in_work = false;
    bool treads_to_work;
    random::random::config random_config;

private:
    u_word iteration = 0;

protected:
    char save_load_size;

public:
    std::unique_ptr<random::random> random_;
    std::vector<std::shared_ptr<neurons::neuron>> storage_;
    std::vector<bool> world_input;
    std::vector<bool> world_output;
    std::vector<bnn::thread> threads;

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
    void fill_threads(u_word threads_count);
    const u_word& get_iteration() const;
    void stop();

private:
    void function();
};

} // namespace bnn

#endif // BNN_BRAIN_H
