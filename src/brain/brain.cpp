/*
 *   Binary Neurons Network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#include "brain.h"

#include <unistd.h>

#include <algorithm>
#include <iostream>

#include "thread.h"
#include "storage.hpp"

namespace bnn
{

brain::~brain()
{
    logging("");
}

brain::brain(u_word quantity_of_neurons_in_power_of_two,
             u_word input_length,
             u_word output_length,
             u_word threads_count_in_power_of_two)
    : quantity_of_neurons_in_power_of_two(quantity_of_neurons_in_power_of_two),
      threads_count(simple_math::two_pow_x(threads_count_in_power_of_two)),
      quantity_of_neurons_sensor(input_length),
      quantity_of_neurons_motor(output_length)
{
    u_word random_array_length_in_power_of_two = quantity_of_neurons_in_power_of_two + 7;

    if(random_array_length_in_power_of_two > QUANTITY_OF_BITS_IN_WORD)
        random_array_length_in_power_of_two = QUANTITY_OF_BITS_IN_WORD;

    quantity_of_neurons = simple_math::two_pow_x(quantity_of_neurons_in_power_of_two);
    quantity_of_neurons_binary = quantity_of_neurons - quantity_of_neurons_sensor - quantity_of_neurons_motor;

    if(quantity_of_neurons <= quantity_of_neurons_sensor + quantity_of_neurons_motor)
        throw_error("quantity_of_neurons_sensor + quantity_of_neurons_motor >= quantity_of_neurons_end");

    random_.reset(new random::random(random_array_length_in_power_of_two));
    storage_.resize(quantity_of_neurons);
    u_word n = 0;
    u_word quantity_of_neurons_per_thread = quantity_of_neurons / threads_count;
    world_input.resize(quantity_of_neurons_sensor);

    auto increment_n = [&]()
    {
        n += quantity_of_neurons_per_thread;

        if(n >= quantity_of_neurons)
            n = n - quantity_of_neurons + 1;
    };

    for(u_word i = 0; i < quantity_of_neurons_sensor; i++)
    {
        world_input[i] = random_.get();
        storage_[n].reset(new neurons::sensor(world_input, i));
        increment_n();
    }

    world_output.resize(quantity_of_neurons_motor);

    for(u_word i = 0; i < quantity_of_neurons_motor; i++)
    {
        world_output[i] = random_.get();
        storage_[n].reset(new neurons::motor(world_output, i));
        increment_n();
    }

    for(u_word i = 0; i < quantity_of_neurons_binary; i++)
    {
        storage_[n].reset(new neurons::binary());
        increment_n();
    }

    u_word random_array_length_per_thread = simple_math::two_pow_x(random_array_length_in_power_of_two)
            / threads_count / QUANTITY_OF_BITS_IN_WORD;

    u_word start_neuron = 0;
    u_word length_in_us_in_power_of_two = simple_math::log2_1(quantity_of_neurons_per_thread);

    for(u_word i = 0; i < threads_count; i++)
    {
        random::config random_config;
        random_config.put_offset_start = random_array_length_per_thread * i;
        random_config.put_offset = random_config.put_offset_start;
        random_config.put_offset_end = random_array_length_per_thread * (i + 1);
        threads.push_back(bnn::thread(this, i, start_neuron, length_in_us_in_power_of_two, random_config));
        start_neuron += quantity_of_neurons_per_thread;
    }

    random_config.put_offset_start = 0;
    random_config.put_offset = random_config.put_offset_start;
    random_config.put_offset_end = simple_math::two_pow_x(random_array_length_in_power_of_two) / QUANTITY_OF_BITS_IN_WORD;
}

const u_word& brain::get_iteration() const
{
    return iteration;
}

bool brain::get_output(u_word offset) const
{
    return world_output[offset];
}

void brain::function(brain* me)
{
    try
    {
        me->in_work = true;
        u_word iteration_old = 0, iteration = 0, quantity_of_initialized_neurons_binary;
        me->treads_to_work = true;

        for(auto& t : me->threads)
            t.start();

        logging("brain started");

        while(me->to_work)
        {
            if(iteration_old < iteration)
            {
                me->candidate_for_kill = me->random_->get(me->quantity_of_neurons_in_power_of_two, me->random_config);
                iteration_old = iteration;
            }

            iteration = 0;
            quantity_of_initialized_neurons_binary = 0;

            std::for_each(me->threads.begin(), me->threads.end(), [&](const thread& t)
            {
                iteration += t.iteration;
                quantity_of_initialized_neurons_binary += t.quantity_of_initialized_neurons_binary;
            });

            me->iteration = iteration / me->threads_count;
            me->quantity_of_initialized_neurons_binary = quantity_of_initialized_neurons_binary;
            usleep(BNN_LITTLE_TIME);
        }
    }
    catch (...)
    {
        logging("error in brain");
    }

    me->treads_to_work = false;

    while(std::any_of(me->threads.begin(), me->threads.end(), [](const thread& t){ return t.in_work; }))
        usleep(BNN_LITTLE_TIME);

    logging("brain stopped");
    me->in_work = false;
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

    main_thread = std::thread(function, this);
    main_thread.detach();
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
