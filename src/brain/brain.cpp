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
#include "storage.h"

namespace bnn
{

brain::~brain()
{
    logging("");

    stop();

    std::for_each(storage_.begin(), storage_.end(), [](const storage& s)
    {
        if(s.neuron_.type_ == neuron::type::motor)
            delete s.motor_.binary_neurons;
    });
}

brain::brain(_word random_array_length_in_power_of_two,
             _word quantity_of_neurons_in_power_of_two,
             _word input_length,
             _word output_length,
             _word threads_count_in_power_of_two)
    : quantity_of_neurons_in_power_of_two(quantity_of_neurons_in_power_of_two),
      threads_count(simple_math::two_pow_x(threads_count_in_power_of_two)),
      quantity_of_neurons_sensor(input_length),
      quantity_of_neurons_motor(output_length),
      random_array_length_in_power_of_two(random_array_length_in_power_of_two)
{
    quantity_of_neurons = simple_math::two_pow_x(quantity_of_neurons_in_power_of_two);

    quantity_of_neurons_binary = quantity_of_neurons - quantity_of_neurons_sensor - quantity_of_neurons_motor;

    if (quantity_of_neurons <= quantity_of_neurons_sensor + quantity_of_neurons_motor)
        throw_error("quantity_of_neurons_sensor + quantity_of_neurons_motor >= quantity_of_neurons_end");

    random_.reset(new random::random(random_array_length_in_power_of_two, m_sequence_));

    storage_.resize(quantity_of_neurons);

    _word n = 0;

    _word quantity_of_neurons_per_thread = quantity_of_neurons / threads_count;

    world_input.resize(quantity_of_neurons_sensor);

    for(_word i = 0; i < quantity_of_neurons_sensor; i++)
    {
        world_input[i] = m_sequence_.next();

        storage_[n].sensor_ = sensor(world_input, i);

        n += quantity_of_neurons_per_thread;

        if(n >= quantity_of_neurons)
            n = n - quantity_of_neurons + 1;
    }

    world_output.resize(quantity_of_neurons_motor);

    for(_word i = 0; i < quantity_of_neurons_motor; i++)
    {
        world_output[i] = m_sequence_.next();

        storage_[n].motor_ = motor(world_output, i);

        n += quantity_of_neurons_per_thread;

        if(n >= quantity_of_neurons)
            n = n - quantity_of_neurons + 1;
    }

    for(_word i = 0; i < quantity_of_neurons_binary; i++)
    {
        if(storage_[n].neuron_.get_type() != neuron::type::sensor &&
                storage_[n].neuron_.get_type() != neuron::type::motor)
            storage_[n].binary_ = binary();

        n += quantity_of_neurons_per_thread;

        if(n >= quantity_of_neurons)
            n = n - quantity_of_neurons + 1;
    }

    _word random_array_length_per_thread = simple_math::two_pow_x(random_array_length_in_power_of_two)
            / threads_count / QUANTITY_OF_BITS_IN_WORD;

    _word start_neuron = 0;

    _word length_in_us_in_power_of_two = simple_math::log2_1(quantity_of_neurons_per_thread);

    for(_word i = 0; i < threads_count; i++)
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

const _word& brain::get_iteration() const
{
    return iteration;
}

bool brain::get_output(_word offset) const
{
    return world_output[offset];
}

void brain::function(brain *me)
{
    try
    {
        _word iteration_old = 0, iteration = 0, quantity_of_initialized_neurons_binary;

        while(state::start != me->state_);

        std::for_each(me->threads.begin(), me->threads.end(), [](thread& t){ t.start(); });

        while(std::any_of(me->threads.begin(), me->threads.end(), [](const thread& t){ return state::started != t.state_; }));

        me->state_ = state::started;

        logging("brain started");

        while(state::started == me->state_)
        {
            if(iteration_old < iteration)
            {
                {
                    _word candidate_for_kill = 0;

                    for(_word i = 0; i < me->quantity_of_neurons_in_power_of_two; i++)
                    {
                        candidate_for_kill <<= 1;

                        candidate_for_kill |= me->random_->get(1, me->random_config);;
                    }

                    me->candidate_for_kill = candidate_for_kill;
                }

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

            usleep(1000);
        }
    }
    catch (...)
    {
        logging("error in brain");
    }

    std::for_each(me->threads.begin(), me->threads.end(), [](thread& t){ t.state_ = state::stop; });

    while(std::any_of(me->threads.begin(), me->threads.end(), [](const thread& t){ return state::stopped != t.state_; }));

    me->state_ = state::stopped;

    logging("brain stopped");
}

void brain::set_input(_word offset, bool value)
{
    world_input[offset] = value;
}

void brain::start()
{
    logging("brain::start() begin");

    if(state::stopped != state_)
        return;

    main_thread = std::thread(function, this);

    main_thread.detach();

    state_ = state::start;

    while(state::started != state_);

    logging("brain::start() end");
}

void brain::stop()
{
    logging("brain::stop() begin");

    if(state::started != state_)
        return;

    state_ = state::stop;

    while(state::stopped != state_);

    logging("brain::stop() end");
}

} // namespace bnn
