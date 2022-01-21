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
    stop();

    for (_word i = 0; i < quantity_of_neurons; i++)
        if(storage_[i].neuron_.neuron_type_ == neuron::neuron_type::neuron_type_neuron)
            delete storage_[i].motor_.binary_neurons;
}

brain::brain(_word random_array_length_in_power_of_two,
             _word quantity_of_neurons_in_power_of_two,
             _word input_length,
             _word output_length,
             _word threads_count_in_power_of_two)
    : quantity_of_neurons_in_power_of_two(quantity_of_neurons_in_power_of_two),
      quantity_of_neurons_sensor(input_length),
      quantity_of_neurons_motor(output_length),
      random_array_length_in_power_of_two(random_array_length_in_power_of_two)
{
    quantity_of_neurons = simple_math::two_pow_x(quantity_of_neurons_in_power_of_two);

    quantity_of_neurons_binary = quantity_of_neurons - quantity_of_neurons_sensor - quantity_of_neurons_motor;

    threads_count = simple_math::two_pow_x(threads_count_in_power_of_two);

    if (quantity_of_neurons <= quantity_of_neurons_sensor + quantity_of_neurons_motor)
        throw ("quantity_of_neurons_sensor + quantity_of_neurons_motor >= quantity_of_neurons_end");

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

void brain::main_function(brain *brain_)
{
    _word iteration, quantity_of_initialized_neurons_binary;

    while(std::any_of(brain_->threads.begin(), brain_->threads.end(), [](const thread& t){ return !t.in_work; }));

    while(state::started != brain_->state_);

    // TODO replace with bnn::random
    std::mt19937 gen;
    std::uniform_int_distribution<> uid = std::uniform_int_distribution<>(0, 1);

    while(state::started == brain_->state_)
    {
        {
            // TODO replace with bnn::random
            _word candidate_for_kill = 0;
            for(_word i = 0; i < brain_->quantity_of_neurons_in_power_of_two; i++)
            {
                candidate_for_kill <<= 1;
                candidate_for_kill |= static_cast<bool>(uid(gen));
            }
            brain_->candidate_for_kill = candidate_for_kill;
        }

        iteration = 0;

        quantity_of_initialized_neurons_binary = 0;

        std::for_each(brain_->threads.begin(), brain_->threads.end(), [&](const thread& t)
        {
            iteration += t.iteration;

            quantity_of_initialized_neurons_binary += t.quantity_of_initialized_neurons_binary;
        });

        brain_->iteration = iteration / brain_->threads.size();

        brain_->quantity_of_initialized_neurons_binary = quantity_of_initialized_neurons_binary;
    }
}

void brain::set_input(_word offset, bool value)
{
    world_input[offset] = value;
}

void brain::start()
{
    std::cout << "void brain::start()" << std::endl;

    if(state::stopped != state_)
        throw "state_ != state::state_stopped";

    state_ = state::start;

    _word quantity_of_neurons_per_thread = simple_math::two_pow_x(quantity_of_neurons_in_power_of_two) / threads_count;

    _word length_in_us_in_power_of_two = simple_math::log2_1(quantity_of_neurons_per_thread);

    _word start_neuron = 0;

    m_sequence m_sequence(QUANTITY_OF_BITS_IN_WORD - 1);

    random_.reset(new random::random(random_array_length_in_power_of_two, m_sequence));

    for(_word i = 0; i < threads_count; i++)
    {
        std::cout << "thrd.push_back(thread(this, i, start_neuron, length_in_us_in_power_of_two));" << std::endl;

        _word random_array_length_per_thread = simple_math::two_pow_x(quantity_of_neurons_in_power_of_two) / threads_count;

        random::config random_config;

        random_config.put_offset_start = random_array_length_per_thread * i;

        random_config.put_offset_end = random_array_length_per_thread * (i + 1);

        threads.push_back(thread(this, i, start_neuron, length_in_us_in_power_of_two, random_config));

        start_neuron += quantity_of_neurons_per_thread;
    }

    if(!std::any_of(storage_.begin(), storage_.end(), [](storage& u)
    {
                    if(u.neuron_.get_type()==neuron::neuron_type_binary)
                    if(u.binary_.get_type_binary()==binary::neuron_binary_type_in_work)
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

    main_thread = std::thread(main_function, this);

    main_thread.detach();

    std::for_each(threads.begin(), threads.end(), [](thread& t) { t.in_work = true; });

    state_ = state::started;
}

void brain::stop()
{
    if(state_ != state::started)
        throw "state != state::state_started";

    state_ = state::stop;

    while(std::any_of(threads.begin(), threads.end(), [](const thread& t){ return t.in_work; }));

    threads.clear();

    state_ = state::stopped;
}

void brain::primary_filling()
{
    for (_word i = 0; i < storage_.size(); i++)
        if(storage_[i].neuron_.get_type()==neuron::neuron_type_binary)
        {
            candidate_for_kill = i;
            break;
        }

    //_word count = world_input.size() * world_input.size();
    //_word count = quantity_of_neurons / 16;
    _word count = quantity_of_neurons_binary;
    std::cout << std::to_string(count) << std::endl;
    _word i = 0;
    _word n = 0;

    _word thread_number;

    //while(true)
    for (int m1 = 0; m1 < 2; m1++)
        for (int m2 = 0; m2 < 2; m2++)
            for(_word j = 0; j < quantity_of_neurons - 1; j++)
            {
                if(storage_[j].neuron_.get_type() != neuron::neuron_type::neuron_type_sensor)
                    continue;

                for(_word k = j + 1; k < quantity_of_neurons; k++)
                {
                    if(storage_[k].neuron_.get_type() != neuron::neuron_type::neuron_type_sensor)
                        continue;

                    {
                        while(true)
                        {
                            if(i >= count)
                            {
                                return;
                            }

                            i++;

                            if(storage_[n].neuron_.get_type() == neuron::neuron_type::neuron_type_binary)
                            {
                                storage_[j].sensor_.out_new = m1;
                                storage_[k].sensor_.out_new = m2;

                                thread_number = n / (quantity_of_neurons / threads_count);

                                storage_[n].binary_.init(*this, thread_number, j, k, storage_);

                                n += quantity_of_neurons / threads_count;

                                if(n >= quantity_of_neurons)
                                    n = n - quantity_of_neurons + 1;

                                break;
                            }

                            n += quantity_of_neurons / threads_count;

                            if(n >= quantity_of_neurons)
                                n = n - quantity_of_neurons + 1;
                        }
                    }
                }
            }

    std::cout << std::to_string(i) << std::endl;
}

} // namespace bnn
