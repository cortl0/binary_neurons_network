/*
 *   Binary Neurons Network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#include "brain_tools.h"

#include <unistd.h>
#include <algorithm>
#include <iostream>
#include <vector>

#include "brain/thread.h"
#include "brain/storage.h"

namespace bnn
{

brain_tools::~brain_tools()
{

}

brain_tools::brain_tools(_word random_array_length_in_power_of_two,
                         _word quantity_of_neurons_in_power_of_two,
                         _word input_length,
                         _word output_length,
                         _word threads_count_in_power_of_two)
    : brain(random_array_length_in_power_of_two,
            quantity_of_neurons_in_power_of_two,
            input_length,
            output_length,
            threads_count_in_power_of_two)
{

}

void recursion(_word num, brain *b, std::string &ss)
{
    ss += "\n";

    if(b->storage_[num].neuron_.get_type() == neuron::type::sensor)
    {
        ss += "sensor | " + std::to_string(num);
    }

    if(b->storage_[num].neuron_.get_type() == neuron::type::motor)
    {
        ss += "motor  | " + std::to_string(num);
    }

    if(b->storage_[num].neuron_.get_type() == neuron::type::binary && b->storage_[num].binary_.in_work)
    {
        ss += "binary | " + std::to_string(num);
        recursion(b->storage_[num].binary_.first, b, ss);
        recursion(b->storage_[num].binary_.second, b, ss);
    }
};

void brain_tools::debug_out()
{
    thread_debug_out = std::thread([&]()
    {
        try
        {
            while(state::started != state_);

            _word iteration = get_iteration();

            _word old_iteration = iteration;

            while(state::stop != state_)
            {
                usleep(1000);

                iteration = get_iteration();

                if(old_iteration >= iteration)
                    continue;

                if(get_iteration() % 128)
                    continue;
#ifdef DEBUG
                unsigned long long int debug_created = 0;
                unsigned long long int debug_killed = 0;
                _word debug_motors_slots_ocupied = 0;
                _word debug_average_level = 0;
                _word debug_max_level = 0;
                _word debug_max_level_binary_num = 0;
                _word debug_average_consensus = 0;
                _word debug_max_consensus = 0;
                _word debug_max_consensus_binary_num = 0;
                _word debug_max_consensus_motor_num = 0;
                unsigned long long int debug_count_get=0;
                unsigned long long int debug_count_put=0;
                long long int debug_sum_put=0;

                debug_count_get += random_config.debug_count_get;
                debug_count_put += random_config.debug_count_put;
                debug_sum_put += random_config.debug_sum_put;
#endif

                std::for_each(threads.begin(), threads.end(), [&](const thread& t)
                {
#ifdef DEBUG
                    debug_created += t.debug_created;
                    debug_killed += t.debug_killed;

                    debug_average_consensus += t.debug_average_consensus;

                    if(debug_max_consensus < t.debug_max_consensus)
                    {
                        debug_max_consensus = t.debug_max_consensus;
                        debug_max_consensus_binary_num = t.debug_max_consensus_binary_num;
                        debug_max_consensus_motor_num = t.debug_max_consensus_motor_num;
                    }

                    debug_count_get += t.random_config.debug_count_get;
                    debug_count_put += t.random_config.debug_count_put;
                    debug_sum_put += t.random_config.debug_sum_put;
#endif
                });

#ifdef DEBUG
                debug_average_consensus /= threads_count;

                _word debug_count = 0;

                for (_word i = 0; i < storage_.size(); i++)
                {
                    if(storage_[i].neuron_.get_type() == neuron::type::motor)
                    {
                        debug_motors_slots_ocupied += storage_[i].motor_.binary_neurons->size();
                    }

                    if(neuron::type::binary == storage_[i].neuron_.get_type() && storage_[i].binary_.in_work)
                    {
                        debug_average_level += storage_[i].binary_.level;

                        debug_count++;

                        if(debug_max_level < storage_[i].binary_.level)
                        {
                            debug_max_level = storage_[i].binary_.level;
                            debug_max_level_binary_num = i;
                        }
                    }

                    if(debug_count)
                        debug_average_level /= debug_count;
                }
#endif

                {
                    std::string s("\n");

                    s += "iteration " + std::to_string(get_iteration());
                    s += " | initialized " + std::to_string(quantity_of_initialized_neurons_binary);

#ifdef DEBUG
                    s += " = created " + std::to_string(debug_created);
                    s += " - killed " + std::to_string(debug_killed);
                    s += " | motor_slots_ocupied " + std::to_string(debug_motors_slots_ocupied);
                    s += "\n";
                    s += "level     ";
                    s += " | average " + std::to_string(debug_average_level);
                    s += " | max " + std::to_string(debug_max_level);
                    s += " | max_binary_life " + std::to_string(storage_[debug_max_level_binary_num].neuron_.life_number);
                    s += " | max_binary_num " + std::to_string(debug_max_level_binary_num);
                    s += " | max_binary_calculation_count " + std::to_string(storage_[debug_max_level_binary_num].neuron_.calculation_count);
                    s += "\n";
                    s += "consensus ";
                    s += " | average " + std::to_string(debug_average_consensus);
                    s += " | max " + std::to_string(debug_max_consensus);
                    s += " | max_motor_num " + std::to_string(debug_max_consensus_motor_num);
                    s += " | max_binary_life " + std::to_string(storage_[debug_max_consensus_binary_num].neuron_.life_number);
                    s += " | max_binary_num " + std::to_string(debug_max_consensus_binary_num);
                    s += " | max_binary_calculation_count " + std::to_string(storage_[debug_max_consensus_binary_num].neuron_.calculation_count);

                    s += "\n";
                    //recursion(debug_max_consensus_binary_num, this, s);
                    s += "\n";

                    s += "\n";
                    s += "random    ";
                    s += " | size " + std::to_string(random_->get_array().size() * QUANTITY_OF_BITS_IN_WORD);
                    s += " | count_get " + std::to_string(debug_count_get);
                    s += " | count_put " + std::to_string(debug_count_put);
                    s += " | sum_put " + std::to_string(debug_sum_put);
#endif

                    std::cout << s << std::endl;
                }

                old_iteration = iteration;
            }
        }
        catch (...)
        {
            logging("unknown error");
        }
    });

    thread_debug_out.detach();
}

bool brain_tools::load(std::ifstream& ifs)
{
#if (1)
    if(ifs.is_open())
    {
        ifs.read(reinterpret_cast<char*>(this), brain_save_load_length);

        world_input.resize(quantity_of_neurons_sensor);

        bool b;

        for(_word i = 0; i < quantity_of_neurons_sensor; i++)
        {
            ifs.read(reinterpret_cast<char*>(&b), sizeof(b));
            world_input[i] = b;
        }

        world_output.resize(quantity_of_neurons_motor);

        for(_word i = 0; i < quantity_of_neurons_motor; i++)
        {
            ifs.read(reinterpret_cast<char*>(&b), sizeof(b));
            world_output[i] = b;
        }

        storage_.resize(quantity_of_neurons);

        for(_word i = 0; i < quantity_of_neurons; i++)
            ifs.read(reinterpret_cast<char*>(storage_[i].words), sizeof(storage));

        _word w;

        ifs.read(reinterpret_cast<char*>(&w), sizeof(w));

        auto random_array = random_->get_array();

        random_array.resize(w);

        for(_word j = 0; j < random_array.size(); j++)
        {
            w = random_array[j];
            ifs.read(reinterpret_cast<char*>(&w), sizeof(w));
        }

        ifs.read(reinterpret_cast<char*>(&w), sizeof(w));

        threads_count = w;

        threads.resize(threads_count);

        for(_word i = 0; i < threads_count; i++)
        {
            ifs.read(reinterpret_cast<char*>(&threads[i]), thread_save_load_length);
        }

        return true;
    }
#endif
    return false;
}

void brain_tools::primary_filling()
{
    std::vector<_word> busy_neurons;
    std::vector<std::vector<_word>> free_neurons(threads_count, std::vector<_word>());

    for(_word i = 0; i < storage_.size(); i++)
    {
        if((storage_[i].neuron_.get_type() == neuron::type::sensor
            || storage_[i].neuron_.get_type() == neuron::type::motor)
                || (storage_[i].neuron_.get_type() == neuron::type::binary && storage_[i].binary_.in_work))
            busy_neurons.push_back(i);
        else
            free_neurons[i / (storage_.size() / threads_count)].push_back(i);
    }

    if(busy_neurons.size() < 2)
        return;

    _word i, j, thread_number_counter = 0;

    while(std::any_of(free_neurons.begin(), free_neurons.end(), [](const std::vector<_word> &f){ return f.size(); }))
    {
        if(!free_neurons[thread_number_counter].size())
        {
            thread_number_counter++;

            continue;
        }

        i = random_->get_ft(0, busy_neurons.size(), random_config);
        j = random_->get_ft(0, busy_neurons.size(), random_config);

        if(i == j)
            continue;

        storage_[i].neuron_.out_new = random_->get(1, random_config);
        storage_[j].neuron_.out_new = random_->get(1, random_config);
        storage_[i].neuron_.out_old = !storage_[i].neuron_.out_new;
        storage_[j].neuron_.out_old = !storage_[j].neuron_.out_new;

        storage_[free_neurons[thread_number_counter].back()].binary_.init(*this, thread_number_counter, i, j, storage_);

        free_neurons[thread_number_counter].pop_back();

        thread_number_counter++;

        if(thread_number_counter >= threads_count)
            thread_number_counter = 0;
    }
}

void brain_tools::resize(_word brainBits_)
{
    if(state_ != bnn::state::stopped)
        throw_error("brain is running now");

    if(brainBits_ > quantity_of_neurons_in_power_of_two)
    {
        _word quantity_of_neuron_end_temp = 1 << (brainBits_);

        std::vector<bnn::storage> us_temp = std::vector<storage>(quantity_of_neuron_end_temp);

        for(_word i = 0; i < quantity_of_neurons; i++)
            for(_word j = 0; j < sizeof(storage) / sizeof(_word); j++)
                us_temp[i].words[j] = storage_[i].words[j];

        for (_word i = quantity_of_neurons; i < quantity_of_neuron_end_temp; i++)
            us_temp[i].binary_ = binary();

        std::swap(storage_, us_temp);

        quantity_of_neurons_in_power_of_two = brainBits_;
        quantity_of_neurons = quantity_of_neuron_end_temp;
        quantity_of_neurons_binary = quantity_of_neurons - quantity_of_neurons_sensor - quantity_of_neurons_motor;
    }
}

bool brain_tools::save(std::ofstream& ofs)
{
#if (1)
    if(ofs.is_open())
    {
        ofs.write(reinterpret_cast<char*>(this), brain_save_load_length);

        bool b;

        for(_word i = 0; i < quantity_of_neurons_sensor; i++)
        {
            b = world_input[i];
            ofs.write(reinterpret_cast<char*>(&b), sizeof(b));
        }

        for(_word i = 0; i < quantity_of_neurons_motor; i++)
        {
            b = world_output[i];
            ofs.write(reinterpret_cast<char*>(&b), sizeof(b));
        }

        for(_word i = 0; i < quantity_of_neurons; i++)
            ofs.write(reinterpret_cast<char*>(storage_[i].words), sizeof(storage));

        auto random_array = random_->get_array();

        _word w = random_array.size();

        ofs.write(reinterpret_cast<char*>(&w), sizeof(w));

        for(_word j = 0; j < random_array.size(); j++)
        {
            w = random_array[j];
            ofs.write(reinterpret_cast<char*>(&w), sizeof(w));
        }

        w = threads.size();

        ofs.write(reinterpret_cast<char*>(&w), sizeof(w));

        for(_word i = 0; i < threads_count; i++)
        {
            ofs.write(reinterpret_cast<char*>(&threads[i]), thread_save_load_length);

            threads[i].brain_ = this;
        }

        return true;
    }
#endif
    return false;
}

void brain_tools::stop()
{
    brain::stop();
}

} // namespace bnn
