/*
 *   Binary Neurons Network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#include "brain_tools.h"

#include <unistd.h>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <map>
#include <set>
#include <vector>

#include "brain/thread.h"
#include "brain/storage.hpp"

namespace bnn
{

brain_tools::~brain_tools()
{
    logging("");
}

brain_tools::brain_tools(u_word quantity_of_neurons_in_power_of_two,
                         u_word input_length,
                         u_word output_length,
                         u_word threads_count_in_power_of_two)
    : brain(quantity_of_neurons_in_power_of_two,
            input_length,
            output_length,
            threads_count_in_power_of_two)
{

}

void recursion(u_word num, brain* b, std::string& s)
{
    s += "\n";

    switch (b->storage_[num]->get_type())
    {
    case neurons::neuron::type::sensor:
        s += "sensor | " + std::to_string(num);
        break;
    case neurons::neuron::type::motor:
        s += "motor  | " + std::to_string(num);
        break;
    case neurons::neuron::type::binary:
        if(((neurons::binary*)(b->storage_[num].get()))->in_work)
        {
            s += "binary | " + std::to_string(num);
            recursion(((neurons::binary*)(b->storage_[num].get()))->first_input_address, b, s);
            recursion(((neurons::binary*)(b->storage_[num].get()))->second_input_address, b, s);
        }
        break;
    default:
        break;
    }
};

void brain_tools::debug_out()
{
    thread_debug_out = std::thread([&]()
    {
        try
        {
            while(!in_work)
                usleep(BNN_LITTLE_TIME);

            u_word iteration = get_iteration();

            u_word old_iteration = iteration;

            u_word debug_average_level_counter;

            while(in_work)
            {
                iteration = get_iteration();

                if(old_iteration >= iteration)
                    continue;

                if(iteration % 128)
                    continue;

                old_iteration = iteration;

#ifdef DEBUG
                unsigned long long int debug_created = 0;
                unsigned long long int debug_killed = 0;
                u_word debug_motors_slots_ocupied = 0;
                u_word debug_average_level = 0;
                u_word debug_max_level = 0;
                u_word debug_max_level_binary_num = 0;
                u_word debug_average_consensus = 0;
                s_word debug_max_consensus = 0;
                u_word debug_max_consensus_binary_num = 0;
                u_word debug_max_consensus_motor_num = 0;
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

                debug_average_level_counter = 0;

                for (u_word i = 0; i < storage_.size(); i++)
                {
                    switch (storage_[i]->get_type())
                    {
                    case neurons::neuron::type::motor:
                        debug_motors_slots_ocupied += ((neurons::motor*)(storage_[i].get()))->binary_neurons_.size();
                        break;
                    case neurons::neuron::type::binary:
                        if(((neurons::binary*)(storage_[i].get()))->in_work)
                        {
                            debug_average_level += storage_[i]->level;

                            debug_average_level_counter++;

                            if(debug_max_level < storage_[i]->level)
                            {
                                debug_max_level = storage_[i]->level;
                                debug_max_level_binary_num = i;
                            }
                        }
                        break;
                    default:
                        break;
                    }
                }

                if(debug_average_level_counter)
                    debug_average_level /= debug_average_level_counter;
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
                    s += " | max_binary_life " + std::to_string(storage_[debug_max_level_binary_num]->life_counter);
                    s += " | max_binary_num " + std::to_string(debug_max_level_binary_num);
                    s += " | max_binary_calculation_count " + std::to_string(storage_[debug_max_level_binary_num]->calculation_count);
                    s += "\n";
                    s += "consensus ";
                    s += " | average " + std::to_string(debug_average_consensus);
                    s += " | max " + std::to_string(debug_max_consensus);
                    s += " | max_motor_num " + std::to_string(debug_max_consensus_motor_num);
                    s += " | max_binary_life " + std::to_string(storage_[debug_max_consensus_binary_num]->life_counter);
                    s += " | max_binary_num " + std::to_string(debug_max_consensus_binary_num);
                    s += " | max_binary_calculation_count " + std::to_string(storage_[debug_max_consensus_binary_num]->calculation_count);

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

                usleep(BNN_LITTLE_TIME);
            }
        }
        catch (...)
        {
            logging("unknown error");
        }
    });

    thread_debug_out.detach();
}

const u_word& brain_tools::get_iteration() const
{
    return brain::get_iteration();
}

bool brain_tools::load(std::ifstream& ifs)
{
#if (1)
    if(ifs.is_open())
    {
        ifs.read(reinterpret_cast<char*>(this), (char*)&save_load_size - (char*)this);

        world_input.resize(quantity_of_neurons_sensor);
        bool b;

        for(u_word i = 0; i < quantity_of_neurons_sensor; i++)
        {
            ifs.read(reinterpret_cast<char*>(&b), sizeof(b));
            world_input[i] = b;
        }

        world_output.resize(quantity_of_neurons_motor);

        for(u_word i = 0; i < quantity_of_neurons_motor; i++)
        {
            ifs.read(reinterpret_cast<char*>(&b), sizeof(b));
            world_output[i] = b;
        }

        storage_.resize(quantity_of_neurons);
        u_word w;
        neurons::motor::binary_neuron binary_neuron_(0, 0, 0);
        char temp[sizeof(storage)];
        storage* stor = (storage*)temp;

        for(u_word i = 0; i < quantity_of_neurons; i++)
        {
            ifs.read(temp, sizeof(storage));

            switch(stor->neuron_.get_type())
            {
            case neurons::neuron::type::binary:
            {
                storage_[i].reset(new neurons::binary());
                std::memcpy(static_cast<void*>(storage_[i].get()), temp, sizeof(neurons::binary));
                break;
            }
            case neurons::neuron::type::sensor:
            {
                storage_[i].reset(new neurons::sensor(world_input, stor->sensor_.world_input_address));
                std::memcpy(static_cast<void*>(storage_[i].get()), temp, sizeof(neurons::sensor));
                break;
            }
            case neurons::neuron::type::motor:
            {
                storage_[i].reset(new neurons::motor(world_output, stor->motor_.world_output_address));
                std::memcpy(static_cast<void*>(storage_[i].get()), temp,
                            reinterpret_cast<char*>(&(dynamic_cast<neurons::motor*>(storage_[i].get()))->save_load_size)
                            - reinterpret_cast<char*>(dynamic_cast<neurons::motor*>(storage_[i].get())));
                ifs.read(reinterpret_cast<char*>(&w), sizeof(w));

                for(u_word j = 0; j < w; j++)
                {
                    ifs.read((char*)(&binary_neuron_), sizeof(binary_neuron_));

                    ((neurons::motor*)(storage_[i].get()))->binary_neurons_.insert(
                                std::pair<u_word, neurons::motor::binary_neuron>(binary_neuron_.address, binary_neuron_));
                };

                break;
            }
            default:
                break;
            }
        }

        ifs.read(reinterpret_cast<char*>(&w), sizeof(w));

        auto random_array = random_->get_array();

        random_array.resize(w);

        for(u_word j = 0; j < random_array.size(); j++)
            ifs.read(reinterpret_cast<char*>(&random_array[j]), sizeof(u_word));

        ifs.read(reinterpret_cast<char*>(&threads_count), sizeof(threads_count));

        threads.resize(threads_count);

        for(u_word i = 0; i < threads_count; i++)
        {
            ifs.read(reinterpret_cast<char*>(&threads[i]),
                     reinterpret_cast<char*>(&threads[i].save_load_size) -
                     reinterpret_cast<char*>(&threads[i]));
        }

        return true;
    }
#endif
    return false;
}

void brain_tools::primary_filling()
{
    std::vector<u_word> busy_neurons;
    std::vector<std::vector<u_word>> free_neurons(threads_count, std::vector<u_word>());

    for(u_word i = 0; i < storage_.size(); i++)
    {
        if((storage_[i]->get_type() == neurons::neuron::type::sensor
            || storage_[i]->get_type() == neurons::neuron::type::motor)
                || (storage_[i]->get_type() == neurons::neuron::type::binary && (dynamic_cast<neurons::binary*>(storage_[i].get()))->in_work))
            busy_neurons.push_back(i);
        else
            free_neurons[i / (storage_.size() / threads_count)].push_back(i);
    }

    if(busy_neurons.size() < 2)
        return;

    u_word i, j, thread_number_counter = 0;

    while(std::any_of(free_neurons.begin(), free_neurons.end(), [](const std::vector<u_word> &f){ return f.size(); }))
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

        storage_[i]->output_new = random_->get(1, random_config);
        storage_[j]->output_new = random_->get(1, random_config);
        storage_[i]->output_old = !storage_[i]->output_new;
        storage_[j]->output_old = !storage_[j]->output_new;

        ((neurons::binary*)storage_[free_neurons[thread_number_counter].back()].get())->init(*this, thread_number_counter, i, j, storage_);

        free_neurons[thread_number_counter].pop_back();

        thread_number_counter++;

        if(thread_number_counter >= threads_count)
            thread_number_counter = 0;
    }
}

void brain_tools::resize(u_word brainBits_)
{
#if 0
    if(state_ != bnn::state::stopped)
        throw_error("brain is running now");

    if(brainBits_ > quantity_of_neurons_in_power_of_two)
    {
        u_word quantity_of_neuron_end_temp = 1 << (brainBits_);

        std::vector<bnn::storage> us_temp = std::vector<storage>(quantity_of_neuron_end_temp);

        for(u_word i = 0; i < quantity_of_neurons; i++)
            for(u_word j = 0; j < sizeof(storage) / sizeof(u_word); j++)
                us_temp[i].words[j] = storage_[i]->words[j];

        for (u_word i = quantity_of_neurons; i < quantity_of_neuron_end_temp; i++)
            us_temp[i].binary_ = neurons::binary();

        std::swap(storage_, us_temp);

        quantity_of_neurons_in_power_of_two = brainBits_;
        quantity_of_neurons = quantity_of_neuron_end_temp;
        quantity_of_neurons_binary = quantity_of_neurons - quantity_of_neurons_sensor - quantity_of_neurons_motor;
    }
#endif
}

bool brain_tools::save(std::ofstream& ofs)
{
#if (1)
    if(ofs.is_open())
    {
        ofs.write(reinterpret_cast<char*>(this), (char*)&save_load_size - (char*)this);

        bool b;

        for(u_word i = 0; i < quantity_of_neurons_sensor; i++)
        {
            b = world_input[i];
            ofs.write(reinterpret_cast<char*>(&b), sizeof(b));
        }

        for(u_word i = 0; i < quantity_of_neurons_motor; i++)
        {
            b = world_output[i];
            ofs.write(reinterpret_cast<char*>(&b), sizeof(b));
        }

        u_word w;

        for(u_word i = 0; i < quantity_of_neurons; i++)
        {
            ofs.write(reinterpret_cast<char*>(storage_[i].get()), sizeof(storage));

            if(neurons::neuron::type::motor == storage_[i]->get_type())
            {
                w = (dynamic_cast<neurons::motor*>(storage_[i].get()))->binary_neurons_.size();

                ofs.write((char*)(&w), sizeof (w));

                for(auto& [_, binary_neuron] : (dynamic_cast<neurons::motor*>(storage_[i].get()))->binary_neurons_)
                    ofs.write((char*)(&binary_neuron), sizeof(binary_neuron));
            }
        }

        auto random_array = random_->get_array();

        w = random_array.size();

        ofs.write(reinterpret_cast<char*>(&w), sizeof(w));

        for(u_word j = 0; j < random_array.size(); j++)
        {
            w = random_array[j];
            ofs.write(reinterpret_cast<char*>(&w), sizeof(w));
        }

        w = threads.size();

        ofs.write(reinterpret_cast<char*>(&w), sizeof(w));

        for(u_word i = 0; i < threads_count; i++)
        {
            ofs.write(reinterpret_cast<char*>(&threads[i]), (char*)&threads[i].save_load_size - (char*)&threads[i]);

            threads[i].brain_ = this;
        }

        return true;
    }
#endif
    return false;
}

void brain_tools::save_random()
{
    save_random_bin();
    //save_random_csv();
    save_random_csv_line();
}

void brain_tools::save_random_bin()
{
    std::ofstream ofs(fs::current_path() / "random.bin", std::ios::binary);

    auto random_array = random_->get_array();

    u_word w;

    for(u_word j = 0; j < random_array.size(); j++)
    {
        w = random_array[j];
        ofs.write(reinterpret_cast<char*>(&w), sizeof(w));
    }

    ofs.close();
}

union converter
{
    u_word w;
    unsigned char c[4];
} conv;

void brain_tools::save_random_csv()
{
    std::ofstream ofs(fs::current_path() / "random.csv", std::ios::binary);

    auto random_array = random_->get_array();

    for(u_word j = 0; j < /*random_array.size()*/1024; j++)
    {
        conv.w = random_array[j];

        ofs << (int)conv.c[0] << "\n";
        ofs << (int)conv.c[1] << "\n";
        ofs << (int)conv.c[2] << "\n";
        ofs << (int)conv.c[3] << "\n";
    }

    ofs.close();
}

void brain_tools::save_random_csv_line()
{
    std::ofstream ofs(fs::current_path() / "random_line.csv", std::ios::binary);

    union converter
    {
        u_word w;
        u_char c[4];
        u_short s[2];
    } conv;

    u_char c;

    auto& random_array = random_->get_array();

    if(0)
    {
        auto s = std::multiset<u_char>();

        for(u_word j = 0; j < /*random_array.size()*/1024; j++)
        {
            conv.w = random_array[j];

            for(int i = 0; i < 4; i++)
                s.insert(conv.c[i]);
        }

        for(auto i : s)
            ofs << (int)i << ";";

        ofs << "\n";
    }

    {
        size_t max_size = random_array.size();// / 4096;

        auto char_map = std::map<u_char, u_int>();
        auto char_vector = std::vector<u_char>(max_size * sizeof(u_word));

//        auto m_short = std::map<u_short, u_int>();

        //if(0)
        for(u_word j = 0; j < max_size; j++)
        {
            conv.w = random_array[j];

            for(int i = 0; i < 4; i++)
            {
                auto it = char_map.find(conv.c[i]);

                if(it == char_map.end())
                    char_map.insert(std::pair(conv.c[i], 1));
                else
                    it->second++;

                char_vector[j * sizeof(u_word) + i] = conv.c[i];
            }

//            for(int i = 0; i < 2; i++)
//            {
//                auto it = m_short.find(conv.s[i]);

//                if(it == m_short.end())
//                    m_short.insert(std::make_pair(conv.s[i], 1));
//                else
//                    it->second++;
//            }
        }

        size_t counter_c = 0;

        //int counter_s = 0;

        auto c = char_map.begin();

//        auto s = m_short.begin();

        while(counter_c < max_size * sizeof(u_word))//c != char_map.end())// || s != m_short.end())
        {
//            if(s != m_short.end())
//                ofs << counter_s++ << ";" << (int)s++->second << ";";


            ofs << counter_c << ";" << (int)char_vector[counter_c] << ";";

            if(counter_c < char_map.size() && c != char_map.end())
                ofs << (int)c++->second << ";";

            ofs << "\n";

            counter_c++;

            if(counter_c>=256)
                break;
        }
    }

    ofs.close();
}

void brain_tools::stop()
{
    brain::stop();
}

} // namespace bnn
