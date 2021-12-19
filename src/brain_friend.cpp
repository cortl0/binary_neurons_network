/*
 *   binary neurons network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#include "brain_friend.h"

#include <algorithm>
#include <iostream>

#include "brain/thread.h"
#include "brain/storage.h"

namespace bnn
{

brain_friend::brain_friend(bnn::brain &brain_) : brain_(brain_)
{
    brain_.debug_out = debug_out;
}

void recursion(_word num, brain *b, std::string &ss)
{
    ss += "\n";

    if(b->storage_[num].neuron_.get_type() == neuron::neuron_type_sensor)
    {
        ss += "sensor | " + std::to_string(num);
    }

    if(b->storage_[num].neuron_.get_type() == neuron::neuron_type_motor)
    {
        ss += "motor  | " + std::to_string(num);
    }

    if(b->storage_[num].neuron_.get_type() == neuron::neuron_type_binary &&
        b->storage_[num].binary_.get_type_binary() == binary::neuron_binary_type_in_work)
    {
        ss += "binary | " + std::to_string(num);
        recursion(b->storage_[num].binary_.first, b, ss);
        recursion(b->storage_[num].binary_.second, b, ss);
    }
};

void brain_friend::debug_out(brain* brain_, _word &old_iteration)
{
    _word iteration, quantity_of_initialized_neurons_binary;

    while(brain_->state_ != state::stop)
    {
        iteration = 0;

        quantity_of_initialized_neurons_binary = 0;

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
#endif

        std::for_each(brain_->threads.begin(), brain_->threads.end(), [&](const thread& t)
        {
            iteration += t.iteration;
            quantity_of_initialized_neurons_binary += t.quantity_of_initialized_neurons_binary;
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

            //            debug_count_get += t.rndm->debug_count_get;
            //            debug_count_put += t.rndm->debug_count_put;
            //            debug_sum_put += t.rndm->debug_sum_put;
#endif
        });

#ifdef DEBUG
        debug_average_consensus /= brain_->threads_count;

        _word debug_count = 0;

        for (_word i = 0; i < brain_->storage_.size(); i++)
        {
            if(brain_->storage_[i].neuron_.get_type() == neuron::neuron_type_motor)
            {
                debug_motors_slots_ocupied += brain_->storage_[i].motor_.binary_neurons->size();
            }

            if(brain_->storage_[i].neuron_.get_type() == neuron::neuron_type_binary &&
                    brain_->storage_[i].binary_.get_type_binary() == binary::neuron_binary_type_in_work)
            {
                debug_average_level += brain_->storage_[i].binary_.level;

                debug_count++;

                if(debug_max_level < brain_->storage_[i].binary_.level)
                {
                    debug_max_level = brain_->storage_[i].binary_.level;
                    debug_max_level_binary_num = i;
                }
            }

            if(debug_count)
                debug_average_level /= debug_count;
        }
#endif

        brain_->quantity_of_initialized_neurons_binary = quantity_of_initialized_neurons_binary;

        old_iteration = brain_->iteration;

        brain_->iteration = iteration / brain_->threads.size();

        if(old_iteration < brain_->iteration && !(brain_->iteration % 128))
        {
            std::string s("\n");

            s += "iteration " + std::to_string(brain_->iteration);
            s += " | initialized " + std::to_string(brain_->quantity_of_initialized_neurons_binary);

#ifdef DEBUG
            s += " = created " + std::to_string(debug_created);
            s += " - killed " + std::to_string(debug_killed);
            s += " | motor_slots_ocupied " + std::to_string(debug_motors_slots_ocupied);
            s += "\n";
            s += "level     ";
            s += " | average " + std::to_string(debug_average_level);
            s += " | max " + std::to_string(debug_max_level);
            s += " | max_binary_life " + std::to_string(brain_->storage_[debug_max_level_binary_num].neuron_.life_number);
            s += " | max_binary_num " + std::to_string(debug_max_level_binary_num);
            s += " | max_binary_calculation_count " + std::to_string(brain_->storage_[debug_max_level_binary_num].neuron_.calculation_count);
            s += "\n";
            s += "consensus ";
            s += " | average " + std::to_string(debug_average_consensus);
            s += " | max " + std::to_string(debug_max_consensus);
            s += " | max_motor_num " + std::to_string(debug_max_consensus_motor_num);
            s += " | max_binary_life " + std::to_string(brain_->storage_[debug_max_consensus_binary_num].neuron_.life_number);
            s += " | max_binary_num " + std::to_string(debug_max_consensus_binary_num);
            s += " | max_binary_calculation_count " + std::to_string(brain_->storage_[debug_max_consensus_binary_num].neuron_.calculation_count);

            s += "\n";
            recursion(debug_max_consensus_binary_num, brain_, s);
            s += "\n";

            s += "\n";
            s += "random    ";
            s += " | count_get " + std::to_string(debug_count_get);
            s += " | count_put " + std::to_string(debug_count_put);
            s += " | sum_put " + std::to_string(debug_sum_put);
#endif

            std::cout << s << std::endl;
        }
    }
}

bool brain_friend::load(std::ifstream& ifs)
{
#if (1)
    if(ifs.is_open())
    {
        ifs.read(reinterpret_cast<char*>(&brain_.quantity_of_neurons_in_power_of_two), sizeof (brain_.quantity_of_neurons_in_power_of_two));
        ifs.read(reinterpret_cast<char*>(&brain_.quantity_of_neurons), sizeof (brain_.quantity_of_neurons));
        ifs.read(reinterpret_cast<char*>(&brain_.quantity_of_neurons_binary), sizeof (brain_.quantity_of_neurons_binary));
        ifs.read(reinterpret_cast<char*>(&brain_.quantity_of_neurons_sensor), sizeof (brain_.quantity_of_neurons_sensor));
        ifs.read(reinterpret_cast<char*>(&brain_.quantity_of_neurons_motor), sizeof (brain_.quantity_of_neurons_motor));
        ifs.read(reinterpret_cast<char*>(&brain_.iteration), sizeof (brain_.iteration));
        ifs.read(reinterpret_cast<char*>(&brain_.quantity_of_initialized_neurons_binary), sizeof (brain_.quantity_of_initialized_neurons_binary));
        ifs.read(reinterpret_cast<char*>(&brain_.candidate_for_kill), sizeof (brain_.candidate_for_kill));
        ifs.read(reinterpret_cast<char*>(&brain_.state_), sizeof (brain_.state_));
        ifs.read(reinterpret_cast<char*>(&brain_.threads_count), sizeof (brain_.threads_count));

        brain_.world_input.resize(brain_.quantity_of_neurons_sensor);

        bool b;

        for(_word i = 0; i < brain_.quantity_of_neurons_sensor; i++)
        {
            ifs.read(reinterpret_cast<char*>(&b), sizeof (b));
            brain_.world_input[i] = b;
        }

        brain_.world_output.resize(brain_.quantity_of_neurons_motor);

        for(_word i = 0; i < brain_.quantity_of_neurons_motor; i++)
        {
            ifs.read(reinterpret_cast<char*>(&b), sizeof (b));
            brain_.world_output[i] = b;
        }

        brain_.storage_.resize(brain_.quantity_of_neurons);

        _word w;

        for(_word i = 0; i < brain_.quantity_of_neurons; i++)
            for(_word j = 0; j < sizeof(storage) / sizeof(_word); j++)
            {
                ifs.read(reinterpret_cast<char*>(&w), sizeof (w));
                brain_.storage_[i].words[j] = w;
            }

        for(_word i = 0; i < brain_.threads_count; i++)
        {
            _word rndmLength;

            ifs.read(reinterpret_cast<char*>(&rndmLength), sizeof (rndmLength));

//            if(rndmLength != brain_.threads[i].rndm->get_length())
//            {
//                m_sequence m_sequence(_word_bits - 1);
//                brain_.threads[i].rndm.reset(new random::random_put_get(rndmLength, m_sequence));
//            }

//            for(_word j = 0; j < rndmLength; j++)
//            {
//                ifs.read(reinterpret_cast<char*>(&w), sizeof (w));
//                brain_.threads[i].rndm->get_array()[j] = w;
//            }
        }

        //#ifdef DEBUG
        //        ifs.read(reinterpret_cast<char*>(&brain_.rndm->debug_count_put), sizeof (brain_.rndm->debug_count_put));
        //        ifs.read(reinterpret_cast<char*>(&brain_.rndm->debug_count_get), sizeof (brain_.rndm->debug_count_get));
        //#endif

        return true;
    }
#endif
    return false;
}

void brain_friend::resize(_word brainBits_)
{
    if(brain_.state_ != bnn::state::stopped)
        throw "brain is running now";

    if(brainBits_ > brain_.quantity_of_neurons_in_power_of_two)
    {
        _word quantity_of_neuron_end_temp = 1 << (brainBits_);

        std::vector<bnn::storage> us_temp = std::vector<storage>(quantity_of_neuron_end_temp);

        for(_word i = 0; i < brain_.quantity_of_neurons; i++)
            for(_word j = 0; j < sizeof(storage) / sizeof(_word); j++)
                us_temp[i].words[j] = brain_.storage_[i].words[j];

        for (_word i = brain_.quantity_of_neurons; i < quantity_of_neuron_end_temp; i++)
            us_temp[i].binary_ = binary();

        std::swap(brain_.storage_, us_temp);

        brain_.quantity_of_neurons_in_power_of_two = brainBits_;
        brain_.quantity_of_neurons = quantity_of_neuron_end_temp;
        brain_.quantity_of_neurons_binary = brain_.quantity_of_neurons - brain_.quantity_of_neurons_sensor - brain_.quantity_of_neurons_motor;
    }
}

bool brain_friend::save(std::ofstream& ofs)
{
#if (1)
    if(ofs.is_open())
    {
        ofs.write(reinterpret_cast<char*>(&brain_.quantity_of_neurons_in_power_of_two), sizeof (brain_.quantity_of_neurons_in_power_of_two));
        ofs.write(reinterpret_cast<char*>(&brain_.quantity_of_neurons), sizeof (brain_.quantity_of_neurons));
        ofs.write(reinterpret_cast<char*>(&brain_.quantity_of_neurons_binary), sizeof (brain_.quantity_of_neurons_binary));
        ofs.write(reinterpret_cast<char*>(&brain_.quantity_of_neurons_sensor), sizeof (brain_.quantity_of_neurons_sensor));
        ofs.write(reinterpret_cast<char*>(&brain_.quantity_of_neurons_motor), sizeof (brain_.quantity_of_neurons_motor));
        ofs.write(reinterpret_cast<char*>(&brain_.iteration), sizeof (brain_.iteration));
        ofs.write(reinterpret_cast<char*>(&brain_.quantity_of_initialized_neurons_binary), sizeof (brain_.quantity_of_initialized_neurons_binary));
        ofs.write(reinterpret_cast<char*>(&brain_.candidate_for_kill), sizeof (brain_.candidate_for_kill));
        ofs.write(reinterpret_cast<char*>(&brain_.state_), sizeof (brain_.state_));
        ofs.write(reinterpret_cast<char*>(&brain_.threads_count), sizeof (brain_.threads_count));

        bool b;

        for(_word i = 0; i < brain_.quantity_of_neurons_sensor; i++)
        {
            b = brain_.world_input[i];
            ofs.write(reinterpret_cast<char*>(&b), sizeof (b));
        }

        for(_word i = 0; i < brain_.quantity_of_neurons_motor; i++)
        {
            b = brain_.world_output[i];
            ofs.write(reinterpret_cast<char*>(&b), sizeof (b));
        }

        _word w;

        for(_word i = 0; i < brain_.quantity_of_neurons; i++)
            for(_word j = 0; j < sizeof(storage) / sizeof(_word); j++)
            {
                w = brain_.storage_[i].words[j];
                ofs.write(reinterpret_cast<char*>(&w), sizeof (w));
            }

        for(_word i = 0; i < brain_.threads_count; i++)
        {
//            _word rndmLength = brain_.threads[i].rndm->get_length();

//            ofs.write(reinterpret_cast<char*>(&rndmLength), sizeof (rndmLength));

//            for(_word j = 0; j < rndmLength; j++)
//            {
//                w = brain_.threads[i].rndm->get_array()[j];
//                ofs.write(reinterpret_cast<char*>(&w), sizeof (w));
//            }
        }
        //#ifdef DEBUG
        //        ofs.write(reinterpret_cast<char*>(&brain_.rndm->debug_count_put), sizeof (brain_.rndm->debug_count_put));
        //        ofs.write(reinterpret_cast<char*>(&brain_.rndm->debug_count_get), sizeof (brain_.rndm->debug_count_get));
        //#endif

        return true;
    }
#endif
    return false;
}

void brain_friend::stop()
{
    brain_.stop();
}

} // namespace bnn
