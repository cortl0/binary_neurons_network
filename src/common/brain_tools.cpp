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

#include "common/logger.h"

namespace bnn
{

brain_tools::~brain_tools()
{
    logging("");
}

brain_tools::brain_tools(const bnn_settings& bs)
    : architecture(bs)
{
}

void recursion(u_word num, bnn_bnn* b, std::string& s)
{
    s += "\n";

    switch (b->storage_.data[num].neuron_.type_)
    {
    case bnn_neuron::type::sensor:
        s += "sensor | " + std::to_string(num);
        break;
    case bnn_neuron::type::motor:
        s += "motor  | " + std::to_string(num);
        break;
    case bnn_neuron::type::binary:
        if(b->storage_.data[num].binary_.in_work)
        {
            s += "binary | " + std::to_string(num);
            recursion(b->storage_.data[num].binary_.first.address, b, s);
            recursion(b->storage_.data[num].binary_.second.address, b, s);
        }
        break;
    default:
        break;
    }
};

void brain_tools::get_debug_string(std::string& s)
{
    static u_word debug_average_level_counter;

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
#endif

    for(u_word i = 0; i < bnn->threads_.size; ++i)
    {
        const auto t = bnn->threads_.data[i];

#ifdef DEBUG
        debug_created += t.debug_.created;
        debug_killed += t.debug_.killed;

        debug_average_consensus += t.debug_.consensus_.average;

        if(debug_max_consensus < t.debug_.consensus_.max)
        {
            debug_max_consensus = t.debug_.consensus_.max;
            debug_max_consensus_binary_num = t.debug_.consensus_.max_binary_num;
            debug_max_consensus_motor_num = t.debug_.max_consensus_motor_num;
        }

//        debug_count_get += t.random_config.debug_.random_.count_get;
//        debug_count_put += t.random_config.debug_.random_.count_put;
//        debug_sum_put += t.random_config.debug_.random_.sum_put;
#endif
    }

#ifdef DEBUG
    debug_average_consensus /= bnn->threads_.size;

    debug_average_level_counter = 0;
#if(0)
    auto levels = std::vector<u_word>(50, 0);
    auto life_counters = std::vector<u_word>(100, 0);

    for(u_word i = 0; i < bnn->storage_.size; ++i)
    {
        switch(bnn->storage_.data[i].neuron_.type_)
        {
        case bnn_neuron::type::motor:
        {
            //debug_motors_slots_ocupied += get_motor(storage_, i)->binary_neurons.size();
            debug_motors_slots_ocupied += bnn->motor_binaries_.size_per_motor;
            break;
        }
        case bnn_neuron::type::binary:
        {
            if(bnn->storage_.data[i].binary_.in_work)
            {
                debug_average_level += bnn->storage_.data[i].neuron_.level;
                debug_average_level_counter++;

                if(debug_max_level < bnn->storage_.data[i].neuron_.level)
                {
                    debug_max_level = bnn->storage_.data[i].neuron_.level;
                    debug_max_level_binary_num = i;
                }

                if(levels.size() > bnn->storage_.data[i].neuron_.level)
                    levels[bnn->storage_.data[i].neuron_.level]++;
            }

            if(life_counters.size() > bnn->storage_.data[i].neuron_.life_counter)
                life_counters[bnn->storage_.data[i].neuron_.life_counter]++;

            break;
        }
        default:
            break;
        }
    }

    s += "levels: ";
    for(u_word i = 0; i < 50; i++)
    {
        s += std::to_string(levels[i]);
        s += " ";
    }
    s += "\n";

    s += "life_counters: ";
    for(u_word i = 0; i < 100; i++)
    {
        s += std::to_string(life_counters[i]);
        s += " ";
    }
    s += "\n";
    if(debug_average_level_counter)
        debug_average_level /= debug_average_level_counter;
#endif
#endif

    {
        s += "iter " + std::to_string(bnn->parameters_.iteration);
        s += " | bits " + std::to_string(bnn->storage_.size_in_power_of_two);
        s += " | init " + std::to_string(bnn->parameters_.quantity_of_initialized_neurons_binary);
        s += " | created " + std::to_string(bnn->debug_.created);
        s += " | killed " + std::to_string(bnn->debug_.killed);
        s += "\n";
        s += "calculation_count_max " + std::to_string(bnn->debug_.neuron_.calculation_count_max);
        s += " | calculation_count_min " + std::to_string(bnn->debug_.neuron_.calculation_count_min);

#ifdef DEBUG
#if(0)
        s += " = created " + std::to_string(debug_created);
        s += " - killed " + std::to_string(debug_killed);
        s += " | motor_slots_ocupied " + std::to_string(debug_motors_slots_ocupied);
        s += "\n";
        s += "level     ";
        s += " | average " + std::to_string(debug_average_level);
        s += " | max " + std::to_string(debug_max_level);
        s += " | max_binary_life " + std::to_string(bnn->storage_.data[debug_max_level_binary_num].neuron_.life_counter);
        s += " | max_binary_num " + std::to_string(debug_max_level_binary_num);
        s += " | max_binary_calculation_count " + std::to_string(bnn->storage_.data[debug_max_level_binary_num].neuron_.calculation_count);
        s += "\n";
        s += "consensus ";
        s += " | average " + std::to_string(debug_average_consensus);
        s += " | max " + std::to_string(debug_max_consensus);
        s += " | max_motor_num " + std::to_string(debug_max_consensus_motor_num);
        s += " | max_binary_life " + std::to_string(bnn->storage_.data[debug_max_consensus_binary_num].neuron_.life_counter);
        s += " | max_binary_num " + std::to_string(debug_max_consensus_binary_num);
        s += " | max_binary_calculation_count " + std::to_string(bnn->storage_.data[debug_max_consensus_binary_num].neuron_.calculation_count);

        s += "\n";
        recursion(debug_max_consensus_binary_num, this, s);
        s += "\n";
#endif

        s += "\n";
        s += "random bits " + std::to_string(bnn->random_.size_in_power_of_two);
        s += " | get " + std::to_string(bnn->debug_.random_.count_get);
        s += " | put " + std::to_string(bnn->debug_.random_.count_put);
        s += " | sum_put " + std::to_string(bnn->debug_.random_.sum_put);
        s += "\n";
#endif


        //printf("%du\n", bnn_random_pull(&bnn->random_, 9, &bnn->parameters_.random_config));
                //bnn->threads_.data[0].length_in_us_in_power_of_two);
//        printf("%du\n\n", bnn->threads_.data[bnn->threads_.size - 1].length_in_us_in_power_of_two);
    }
}

const u_word& brain_tools::get_iteration() const
{
    return bnn->parameters_.iteration;
}

bool brain_tools::load(std::ifstream& ifs)
{
#if(0)
    if(!ifs.is_open())
        return false;

    u_word w;
    bool b;
    ifs.read(reinterpret_cast<char*>(this), &save_load_size - reinterpret_cast<char*>(this));
    ifs.read(reinterpret_cast<char*>(&w), sizeof(u_word));
    auto& random_array = random_->get_array();
    random_array.resize(w);

    for(u_word j = 0; j < random_array.size(); j++)
    {
        ifs.read(reinterpret_cast<char*>(&w), sizeof(u_word));
        random_array[j] = w;
    }

    ifs.read(reinterpret_cast<char*>(&w), sizeof(u_word));
    world_input.resize(w);

    for(u_word i = 0; i < w; i++)
    {
        ifs.read(reinterpret_cast<char*>(&b), sizeof(bool));
        world_input[i] = b;
    }

    ifs.read(reinterpret_cast<char*>(&w), sizeof(u_word));
    world_output.resize(w);

    for(u_word i = 0; i < w; i++)
    {
        ifs.read(reinterpret_cast<char*>(&b), sizeof(bool));
        world_output[i] = b;
    }

    storage_.resize(quantity_of_neurons);
    neurons::motor::binary_neuron motor_binary_neuron(0, 0, 0);

    for(u_word i = 0; i < quantity_of_neurons; i++)
    {
        ifs.read(reinterpret_cast<char*>(&w), sizeof(u_word));

        switch(static_cast<neurons::neuron::type>(w))
        {
        case neurons::neuron::type::binary:
        {
            storage_[i].reset(new neurons::binary());
            //neurons::binary::construct(dynamic_cast<neurons::binary*>(storage_[i].get()));
            ifs.read(reinterpret_cast<char*>(storage_[i].get()), sizeof(neurons::binary));
            break;
        }
        case neurons::neuron::type::sensor:
        {
            storage_[i].reset(new neurons::sensor(world_input, 0));
            ifs.read(reinterpret_cast<char*>(storage_[i].get()), sizeof(neurons::sensor));
            break;
        }
        case neurons::neuron::type::motor:
        {
            storage_[i].reset(new neurons::motor(world_output, 0));

            ifs.read(reinterpret_cast<char*>(storage_[i].get()),
                     &(dynamic_cast<neurons::motor*>(storage_[i].get()))->save_load_size -
                     reinterpret_cast<char*>(storage_[i].get()));

            ifs.read(reinterpret_cast<char*>(&w), sizeof(u_word));

            for(u_word j = 0; j < w; j++)
            {
                ifs.read(reinterpret_cast<char*>(&motor_binary_neuron), sizeof(neurons::motor::binary_neuron));

                reinterpret_cast<neurons::motor*>(storage_[i].get())->binary_neurons.insert(
                            std::make_pair(motor_binary_neuron.address, neurons::motor::binary_neuron(motor_binary_neuron)));
            };

            break;
        }
        default:
            break;
        }
    }

    ifs.read(reinterpret_cast<char*>(&w), sizeof(u_word));
    fill_threads(w);

    for(u_word i = 0; i < w; i++)
    {
        ifs.read(reinterpret_cast<char*>(&threads[i].iteration),
                 &threads[i].save_load_size -
                 reinterpret_cast<char*>(&threads[i].iteration));
    }

#endif
    return true;
}

void brain_tools::resize(u_word brainBits_)
{
#if(0)
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
#if(0)
    if(!ofs.is_open())
        return false;

    u_word w;
    bool b;
    ofs.write(reinterpret_cast<char*>(this), &save_load_size - reinterpret_cast<char*>(this));
    auto& random_array = random_->get_array();
    w = random_array.size();
    ofs.write(reinterpret_cast<char*>(&w), sizeof(u_word));

    for(u_word j = 0; j < random_array.size(); j++)
    {
        w = random_array[j];
        ofs.write(reinterpret_cast<char*>(&w), sizeof(u_word));
    }

    w = world_input.size();
    ofs.write(reinterpret_cast<char*>(&w), sizeof(u_word));

    for(u_word i = 0; i < w; i++)
    {
        b = world_input[i];
        ofs.write(reinterpret_cast<char*>(&b), sizeof(bool));
    }

    w = world_output.size();
    ofs.write(reinterpret_cast<char*>(&w), sizeof(u_word));

    for(u_word i = 0; i < w; i++)
    {
        b = world_output[i];
        ofs.write(reinterpret_cast<char*>(&b), sizeof(bool));
    }

    for(u_word i = 0; i < quantity_of_neurons; i++)
    {
        w = static_cast<u_word>(storage_[i]->get_type());
        ofs.write(reinterpret_cast<char*>(&w), sizeof(u_word));

        switch(storage_[i]->get_type())
        {
        case neurons::neuron::type::binary:
            ofs.write(reinterpret_cast<char*>(storage_[i].get()), sizeof(neurons::binary));
            break;
        case neurons::neuron::type::sensor:
            ofs.write(reinterpret_cast<char*>(storage_[i].get()), sizeof(neurons::sensor));
            break;
        case neurons::neuron::type::motor:
            ofs.write(reinterpret_cast<char*>(storage_[i].get()),
                      &(dynamic_cast<neurons::motor*>(storage_[i].get()))->save_load_size -
                      reinterpret_cast<char*>(storage_[i].get()));

            w = (reinterpret_cast<neurons::motor*>(storage_[i].get()))->binary_neurons.size();
            ofs.write(reinterpret_cast<char*>(&w), sizeof(u_word));

            for(auto& binary_neuron : (dynamic_cast<neurons::motor*>(storage_[i].get()))->binary_neurons)
            {
                neurons::motor::binary_neuron motor_binary_neuron(binary_neuron.second);
                ofs.write(reinterpret_cast<char*>(&motor_binary_neuron), sizeof(neurons::motor::binary_neuron));
            }

            break;
        default:
            break;
        }
    }

    w = threads.size();
    ofs.write(reinterpret_cast<char*>(&w), sizeof(u_word));

    for(u_word i = 0; i < w; i++)
    {
        ofs.write(reinterpret_cast<char*>(&threads[i].iteration),
                  &threads[i].save_load_size -
                  reinterpret_cast<char*>(&threads[i].iteration));
    }

#endif
    return true;
}

void brain_tools::save_random()
{
    save_random_bin();
    //save_random_csv();
    save_random_csv_line();
}

void brain_tools::save_random_bin()
{
#if(0)
    std::ofstream ofs(fs::current_path() / "random.bin", std::ios::binary);

    auto random_array = random_->get_array();

    u_word w;

    for(u_word j = 0; j < random_array.size(); j++)
    {
        w = random_array[j];
        ofs.write(reinterpret_cast<char*>(&w), sizeof(w));
    }

    ofs.close();
#endif
}

union converter
{
    u_word w;
    unsigned char c[4];
} conv;

void brain_tools::save_random_csv()
{
#if(0)
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
#endif
}

void brain_tools::save_random_csv_line()
{
#if(0)
    std::ofstream ofs(fs::current_path() / "random_line.csv", std::ios::binary);

    union converter
    {
        u_word w;
        u_char c[4];
        u_short s[2];
    } conv;

    u_char c;

    if(0)
    {
        auto s = std::multiset<u_char>();

        for(u_word j = 0; j < /*random_array.size()*/1024; j++)
        {
            conv.w = bnn->random_.data[j];

            for(int i = 0; i < 4; i++)
                s.insert(conv.c[i]);
        }

        for(auto i : s)
            ofs << (int)i << ";";

        ofs << "\n";
    }

    {
        size_t max_size = bnn->random_.size;// / 4096;

        auto char_map = std::map<u_char, u_int>();
        auto char_vector = std::vector<u_char>(max_size * sizeof(u_word));

//        auto m_short = std::map<u_short, u_int>();

        //if(0)
        for(u_word j = 0; j < max_size; j++)
        {
            conv.w = bnn->random_.data[j];

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
#endif
}

void brain_tools::stop()
{
    architecture::stop();
}

} // namespace bnn
