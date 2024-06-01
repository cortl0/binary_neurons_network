/*
 *   Binary Neurons Network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#include "bnn_tools.h"

#include <algorithm>
#include <map>
#include <vector>

#include "common/logger.h"

namespace bnn
{

bnn_tools::~bnn_tools()
{
    logging("");
}

bnn_tools::bnn_tools(const bnn_settings& bs)
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

void bnn_tools::get_debug_string(std::string& s)
{
    static u_word debug_average_level_counter;

#ifdef DEBUG
    u_word debug_motors_slots_ocupied = 0;
    u_word debug_average_level = 0;
    u_word debug_max_level = 0;
    u_word debug_max_level_binary_num = 0;
#endif

#ifdef DEBUG

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
                ++debug_average_level_counter;

                if(debug_max_level < bnn->storage_.data[i].neuron_.level)
                {
                    debug_max_level = bnn->storage_.data[i].neuron_.level;
                    debug_max_level_binary_num = i;
                }

                if(levels.size() > bnn->storage_.data[i].neuron_.level)
                    ++levels[bnn->storage_.data[i].neuron_.level];
            }

            if(life_counters.size() > bnn->storage_.data[i].neuron_.life_counter)
                ++life_counters[bnn->storage_.data[i].neuron_.life_counter];

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
        s += " | in " + std::to_string(bnn->input_.size);
        s += " | out " + std::to_string(bnn->output_.size);
        s += " | init " + std::to_string(bnn->parameters_.quantity_of_initialized_neurons_binary);
        s += " | created " + std::to_string(bnn->debug_.created);
        s += " | killed " + std::to_string(bnn->debug_.killed);
        s += "\n";
        s += "calculation_count_max " + std::to_string(bnn->debug_.neuron_.calculation_count_max);
        s += " | calculation_count_min " + std::to_string(bnn->debug_.neuron_.calculation_count_min);
        s += "\n";

#ifdef DEBUG
#if(0)
        s += " | motor_slots_ocupied " + std::to_string(debug_motors_slots_ocupied);
        s += "\n";
        s += "level     ";
        s += " | average " + std::to_string(debug_average_level);
        s += " | max " + std::to_string(debug_max_level);
        s += " | max_binary_life " + std::to_string(bnn->storage_.data[debug_max_level_binary_num].neuron_.life_counter);
        s += " | max_binary_num " + std::to_string(debug_max_level_binary_num);
        s += " | max_binary_calculation_count " + std::to_string(bnn->storage_.data[debug_max_level_binary_num].neuron_.calculation_count);
        s += "\n";
#endif
        s += "consensus";
        s += " | average " + std::to_string(bnn->debug_.consensus_.average);
        s += " | max " + std::to_string(bnn->debug_.consensus_.max);

        s += "\n";
        if(bnn->debug_.max_consensus_motor_num != ~u_word{0} && bnn->debug_.consensus_.max_binary_num != ~u_word{0})
        {
            s += " | max_motor_num " + std::to_string(bnn->debug_.max_consensus_motor_num);
            //s += " | max_binary_life " + std::to_string(bnn->storage_.data[bnn->debug_.consensus_.max_binary_num].neuron_.life_counter);
            s += " | max_binary_num " + std::to_string(bnn->debug_.consensus_.max_binary_num);
            //s += " | max_binary_calculation_count " + std::to_string(bnn->storage_.data[bnn->debug_.consensus_.max_binary_num].neuron_.calculation_count);
        }

        s += "\n";
#if(0)

        s += "\n";
        recursion(debug_max_consensus_binary_num, this, s);
        s += "\n";
#endif

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

const u_word& bnn_tools::get_iteration() const
{
    return bnn->parameters_.iteration;
}

bool bnn_tools::load(std::ifstream& ifs)
{
    if(!ifs.is_open())
        return false;

    ifs.read(reinterpret_cast<char*>(&bnn->parameters_.size), sizeof(bnn->parameters_.size));
    ifs.read(reinterpret_cast<char*>(bnn), bnn->parameters_.size);
    calculate_pointers();

    return true;
}

void bnn_tools::resize(u_word brainBits_)
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

bool bnn_tools::save(std::ofstream& ofs)
{
    if(!ofs.is_open())
        return false;

    ofs.write(reinterpret_cast<char*>(&bnn->parameters_.size), sizeof(bnn->parameters_.size));
    ofs.write(reinterpret_cast<char*>(bnn), bnn->parameters_.size);

    return true;
}

void bnn_tools::save_random()
{
    save_random_bin();
    //save_random_csv();
    save_random_csv_line();
}

void bnn_tools::save_random_bin()
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

void bnn_tools::save_random_csv()
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

void bnn_tools::save_random_csv_line()
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
                    ++it->second;

                char_vector[j * sizeof(u_word) + i] = conv.c[i];
            }

//            for(int i = 0; i < 2; i++)
//            {
//                auto it = m_short.find(conv.s[i]);

//                if(it == m_short.end())
//                    m_short.insert(std::make_pair(conv.s[i], 1));
//                else
//                    ++it->second;
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

            ++counter_c;

            if(counter_c>=256)
                break;
        }
    }

    ofs.close();
#endif
}

void bnn_tools::stop()
{
    architecture::stop();
}

} // namespace bnn
