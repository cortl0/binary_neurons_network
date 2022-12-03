/*
 *   Binary Neurons Network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#include "bnn.h"

#include <unistd.h>

#include <algorithm>
#include <iostream>

#include "../common/headers/simple_math.hpp"
#include "neurons/storage.hpp"
#include "thread.h"

namespace bnn
{

brain::~brain()
{
    logging("");
    stop();
}

brain::brain(u_word quantity_of_neurons_in_power_of_two,
             u_word input_length,
             u_word output_length,
             u_word threads_count_in_power_of_two)
    : quantity_of_neurons_in_power_of_two(quantity_of_neurons_in_power_of_two)
{
    const u_word experimental_coefficient = simple_math::log2_1(QUANTITY_OF_BITS_IN_WORD) + 5;

    u_word random_array_length_in_power_of_two = quantity_of_neurons_in_power_of_two + experimental_coefficient;

    if(random_array_length_in_power_of_two >= QUANTITY_OF_BITS_IN_WORD)
    {
        random_array_length_in_power_of_two = QUANTITY_OF_BITS_IN_WORD - 1;
        logging("random_array_length_in_power_of_two does not satisfy of experimental_coefficient");
    }

    if(random_array_length_in_power_of_two > 29)
    {
        random_array_length_in_power_of_two = 29;
        logging("random_array_length_in_power_of_two hardware limitation");
    }

    quantity_of_neurons = simple_math::two_pow_x(quantity_of_neurons_in_power_of_two);
    quantity_of_neurons_binary = quantity_of_neurons - input_length - output_length;

    if(quantity_of_neurons <= input_length + output_length)
        throw_error("quantity_of_neurons_sensor + quantity_of_neurons_motor >= quantity_of_neurons_end");

    random_.reset(new random::random(random_array_length_in_power_of_two));
    storage_.resize(quantity_of_neurons);
    u_word n = 0;

    random_config.put_offset_start = 0;
    random_config.put_offset = random_config.put_offset_start;
    random_config.put_offset_end = simple_math::two_pow_x(random_array_length_in_power_of_two) / QUANTITY_OF_BITS_IN_WORD;

    u_word threads_count = simple_math::two_pow_x(threads_count_in_power_of_two);
    u_word quantity_of_neurons_per_thread = quantity_of_neurons / threads_count;

    world_input.resize(input_length);

    auto increment_n = [&]()
    {
        n += quantity_of_neurons_per_thread;

        if(n >= quantity_of_neurons)
            n = n - quantity_of_neurons + 1;
    };

    for(u_word i = 0; i < input_length; i++)
    {
        world_input[i] = random_->get(1, random_config);
        storage_[n].reset(new neurons::sensor(world_input, i));
        increment_n();
    }

    world_output.resize(output_length);

    for(u_word i = 0; i < output_length; i++)
    {
        world_output[i] = random_->get(1, random_config);
        storage_[n].reset(new neurons::motor(world_output, i));
        increment_n();
    }

    for(u_word i = 0; i < quantity_of_neurons_binary; i++)
    {
        storage_[n].reset(new neurons::binary());
        increment_n();
    }

    fill_threads(threads_count);
}

const u_word& brain::get_iteration() const
{
    return iteration;
}

bool brain::get_output(u_word offset) const
{
    return world_output[offset];
}

void brain::function()
{
    try
    {
        u_word iteration_old = 0, iteration_new = 0, quantity_of_initialized_neurons_binary_temp;
        treads_to_work = true;

        for(auto& t : threads)
            t.start();

        in_work = true;

        logging("brain started");

        while(to_work)
        {
            if(iteration_old < iteration_new)
            {
                candidate_for_kill = random_->get(quantity_of_neurons_in_power_of_two, random_config);
                iteration_old = iteration_new;
            }

            iteration_new = 0;
            quantity_of_initialized_neurons_binary_temp = 0;

            for(const auto& t : threads)
            {
                iteration_new += t.iteration;
                quantity_of_initialized_neurons_binary_temp += t.quantity_of_initialized_neurons_binary;
            }

            iteration = iteration_new / threads.size();
            quantity_of_initialized_neurons_binary = quantity_of_initialized_neurons_binary_temp;
            usleep(BNN_LITTLE_TIME);
        }
    }
    catch (...)
    {
        logging("error in brain");
    }

    treads_to_work = false;

    while(std::any_of(threads.begin(), threads.end(), [](const thread& t){ return t.in_work; }))
        usleep(BNN_LITTLE_TIME);

    logging("brain stopped");
    in_work = false;
}

void brain::set_input(u_word offset, bool value)
{
    world_input[offset] = value;
}

void brain::start()
{
    if(in_work)
        return;

    to_work = true;

    std::thread(&brain::function, this).detach();

    while(!in_work)
        usleep(BNN_LITTLE_TIME);
}

void brain::fill_threads(u_word threads_count)
{
    threads.clear();
    u_word random_array_length_per_thread = random_->get_array().size() / threads_count;
    u_word start_neuron = 0;
    u_word quantity_of_neurons_per_thread = quantity_of_neurons / threads_count;
    u_word length_in_us_in_power_of_two = simple_math::log2_1(quantity_of_neurons_per_thread);

    for(u_word i = 0; i < threads_count; i++)
    {
        random::random::config random_config;
        random_config.put_offset_start = random_array_length_per_thread * i;
        random_config.put_offset = random_config.put_offset_start;
        random_config.put_offset_end = random_array_length_per_thread * (i + 1);
        threads.push_back(bnn::thread(this, i, start_neuron, length_in_us_in_power_of_two, random_config));
        start_neuron += quantity_of_neurons_per_thread;
    }
}

void brain::stop()
{
    if(!in_work)
        return;

    to_work = false;

    while(std::any_of(threads.begin(), threads.end(), [](const thread& t){ return t.in_work; }))
        usleep(BNN_LITTLE_TIME);
}

} // namespace bnn
