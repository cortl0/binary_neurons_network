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
        if(s.neuron_.neuron_type_ == neuron::neuron_type::neuron_type_motor)
            delete s.motor_.binary_neurons;
    });
}

brain::brain(_word random_array_length_in_power_of_two,
             _word quantity_of_neurons_in_power_of_two,
             _word input_length,
             _word output_length,
             _word threads_count_in_power_of_two)
    : quantity_of_neurons_in_power_of_two(quantity_of_neurons_in_power_of_two),
      quantity_of_neurons_sensor(input_length),
      quantity_of_neurons_motor(output_length),
      random_array_length_in_power_of_two(random_array_length_in_power_of_two),
      threads_count(simple_math::two_pow_x(threads_count_in_power_of_two))
{
    quantity_of_neurons = simple_math::two_pow_x(quantity_of_neurons_in_power_of_two);

    quantity_of_neurons_binary = quantity_of_neurons - quantity_of_neurons_sensor - quantity_of_neurons_motor;

    if (quantity_of_neurons <= quantity_of_neurons_sensor + quantity_of_neurons_motor)
        throw_error("quantity_of_neurons_sensor + quantity_of_neurons_motor >= quantity_of_neurons_end");

    storage_.resize(quantity_of_neurons);

    _word n = 0;

    bool ft = false;

    world_input.resize(quantity_of_neurons_sensor);

    for(_word i = 0; i < quantity_of_neurons_sensor; i++)
    {
        world_input[i] = ft = !ft;

        storage_[i].sensor_ = sensor(world_input, i);

        n += quantity_of_neurons / threads_count;

        if(n >= quantity_of_neurons)
            n = n - quantity_of_neurons + 1;
    }

    world_output.resize(quantity_of_neurons_motor);

    for(_word i = 0; i < quantity_of_neurons_motor; i++)
    {
        world_output[i] = ft = !ft;

        storage_[i + quantity_of_neurons_sensor].motor_ = motor(world_output, i);

        n += quantity_of_neurons / threads_count;

        if(n >= quantity_of_neurons)
            n = n - quantity_of_neurons + 1;
    }

    for(_word i = 0; i < quantity_of_neurons_binary; i++)
    {
        if(storage_[n].neuron_.get_type() != neuron::neuron_type::neuron_type_sensor &&
                storage_[n].neuron_.get_type() != neuron::neuron_type::neuron_type_motor)
            storage_[n].binary_ = binary();

        n += quantity_of_neurons / threads_count;

        if(n >= quantity_of_neurons)
            n = n - quantity_of_neurons + 1;
    }
}

const _word& brain::get_input_length() const
{
    return quantity_of_neurons_sensor;
}

const _word& brain::get_iteration() const
{
    return iteration;
}

bool brain::get_output(_word offset) const
{
    return world_output[offset];
}

const _word& brain::get_output_length() const
{
    return quantity_of_neurons_motor;
}

void brain::function(brain *me)
{
    try
    {
        _word iteration, quantity_of_initialized_neurons_binary;

        while(state::start != me->state_);

        std::for_each(me->threads.begin(), me->threads.end(), [](thread& t){ t.start(); });

        while(std::any_of(me->threads.begin(), me->threads.end(), [](const thread& t){ return state::started != t.state_; }));

        me->state_ = state::started;

        logging("brain started");

        while(state::started == me->state_)
        {
            {
                // TODO replace with bnn::random
                static std::mt19937 gen;
                static std::uniform_int_distribution<> uid = std::uniform_int_distribution<>(0, 1);
                _word candidate_for_kill = 0;
                for(_word i = 0; i < me->quantity_of_neurons_in_power_of_two; i++)
                {
                    candidate_for_kill <<= 1;
                    candidate_for_kill |= static_cast<bool>(uid(gen));
                }
                me->candidate_for_kill = candidate_for_kill;
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

void brain::primary_filling()
{
    std::vector<_word> busy_neurons;
    std::vector<_word> free_neurons;
    std::vector<_word> temp;

    for(_word i = 0; i < storage_.size(); i++)
    {
        if((storage_[i].neuron_.get_type() == storage_[i].neuron_.neuron_type_sensor
            || storage_[i].neuron_.get_type() == storage_[i].neuron_.neuron_type_motor)
                || (storage_[i].neuron_.get_type() == storage_[i].neuron_.neuron_type_binary &&
                    storage_[i].binary_.get_type_binary() == binary::neuron_binary_type_in_work))
            busy_neurons.push_back(i);
        else
            free_neurons.push_back(i);
    }

    bool ft = false;

    _word i, j, thread_number;

    while(free_neurons.size())
    {
        j = busy_neurons.size() / 2;

        for(i = 0; i < busy_neurons.size() && free_neurons.size(); i++)
        {
            if(j > busy_neurons.size())
                j = 0;

            storage_[i].sensor_.out_old = !ft;
            storage_[i].sensor_.out_new = ft;
            storage_[j].sensor_.out_old = !ft;
            storage_[j].sensor_.out_new = ft;

            thread_number = free_neurons.back() / (quantity_of_neurons / threads_count);

            storage_[free_neurons.back()].binary_.init(*this, thread_number, i, j, storage_);

            temp.push_back(free_neurons.back());

            free_neurons.pop_back();

            ft = !ft;

            j++;
        }

        std::copy(temp.begin(), temp.end(), std::back_inserter(busy_neurons));

        temp.clear();
    }
}

void brain::set_input(_word offset, bool value)
{
    world_input[offset] = value;
}

void brain::start()
{
    logging("brain::start() begin");

    if(state::stop == state_)
        while(state::stopped != state_);

    if(state::start == state_ || state::started == state_ || state::stopped != state_)
        return;



    //return;

    _word quantity_of_neurons_per_thread = simple_math::two_pow_x(quantity_of_neurons_in_power_of_two) / threads_count;

    _word length_in_us_in_power_of_two = simple_math::log2_1(quantity_of_neurons_per_thread);

    _word start_neuron = 0;

    m_sequence m_sequence(QUANTITY_OF_BITS_IN_WORD - 1);

    random_.reset(new random::random(random_array_length_in_power_of_two, m_sequence));

    for(_word i = 0; i < threads_count; i++)
    {
        _word random_array_length_per_thread = simple_math::two_pow_x(random_array_length_in_power_of_two)
                / threads_count / QUANTITY_OF_BITS_IN_WORD;

        random::config random_config;

        random_config.put_offset_start = random_array_length_per_thread * i;

        random_config.put_offset_end = random_array_length_per_thread * (i + 1);

        threads.push_back(bnn::thread(this, i, start_neuron, length_in_us_in_power_of_two, random_config));

        start_neuron += quantity_of_neurons_per_thread;
    }

    if(!std::any_of(storage_.begin(), storage_.end(), [](const storage& u)
    {
                    if(u.neuron_.get_type() == neuron::neuron_type_binary)
                    if(u.binary_.get_type_binary() == binary::neuron_binary_type_in_work)
                    return true;
                    return false;
}))
    {
        primary_filling();

        iteration = 0;
    }
    else
    {
        for(_word i = 0; i < threads_count; i++)
        {
            threads[i].quantity_of_initialized_neurons_binary = 0;

            for(_word j = 0; j < quantity_of_neurons_per_thread; j++)
                if(storage_[j + threads[i].start_neuron].neuron_.get_type()==neuron::neuron_type_binary)
                    if(storage_[j + threads[i].start_neuron].binary_.get_type_binary()==binary::neuron_binary_type_in_work)
                        threads[i].quantity_of_initialized_neurons_binary++;
        }
    }

    main_thread = std::thread(function, this);

    main_thread.detach();

    state_ = state::start;

    while(state::started != state_);

    logging("brain::start() end");
}

void brain::stop()
{
    logging("brain::stop() begin");

    if(state::start == state_)
        while(state::started != state_);

    if(state::stop == state_ || state::stopped == state_ || state::started != state_)
        return;



    //logging("brain stop_go");



    state_ = state::stop;

    while(state::stopped != state_);

    threads.clear();

    logging("brain::stop() end");
}

} // namespace bnn
