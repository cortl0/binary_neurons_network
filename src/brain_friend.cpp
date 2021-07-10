/*
 *   binary neurons network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#include "brain_friend.h"

namespace bnn
{

brain_friend::brain_friend(bnn::brain &brain_) : brain_(brain_)
{

}

bool brain_friend::load(std::ifstream& ifs)
{
    if(ifs.is_open())
    {
        ifs.read(reinterpret_cast<char*>(&brain_.quantity_of_neurons_in_power_of_two), sizeof (brain_.quantity_of_neurons_in_power_of_two));
        ifs.read(reinterpret_cast<char*>(&brain_.quantity_of_neurons_in_power_of_two_max), sizeof (brain_.quantity_of_neurons_in_power_of_two_max));
        ifs.read(reinterpret_cast<char*>(&brain_.quantity_of_neurons), sizeof (brain_.quantity_of_neurons));
        ifs.read(reinterpret_cast<char*>(&brain_.quantity_of_neurons_binary), sizeof (brain_.quantity_of_neurons_binary));
        ifs.read(reinterpret_cast<char*>(&brain_.quantity_of_neurons_sensor), sizeof (brain_.quantity_of_neurons_sensor));
        ifs.read(reinterpret_cast<char*>(&brain_.quantity_of_neurons_motor), sizeof (brain_.quantity_of_neurons_motor));
        ifs.read(reinterpret_cast<char*>(&brain_.iteration), sizeof (brain_.iteration));
        ifs.read(reinterpret_cast<char*>(&brain_.quantity_of_initialized_neurons_binary), sizeof (brain_.quantity_of_initialized_neurons_binary));

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

        brain_.us.resize(brain_.quantity_of_neurons);

        _word w;

        for(_word i = 0; i < brain_.quantity_of_neurons; i++)
            for(_word j = 0; j < sizeof(brain::union_storage) / sizeof(_word); j++)
            {
                ifs.read(reinterpret_cast<char*>(&w), sizeof (w));
                brain_.us[i].words[j] = w;
            }

        _word rndmLength;

        ifs.read(reinterpret_cast<char*>(&rndmLength), sizeof (rndmLength));

        if(rndmLength != brain_.rndm->get_length())
            brain_.rndm.reset(new random_put_get(rndmLength, 3));

        for(_word i = 0; i < brain_.rndm->get_length(); i++)
        {
            ifs.read(reinterpret_cast<char*>(&w), sizeof (w));
            brain_.rndm->get_array()[i] = w;
        }

        ifs.read(reinterpret_cast<char*>(&brain_.rndm->debug_count_put), sizeof (brain_.rndm->debug_count_put));
        ifs.read(reinterpret_cast<char*>(&brain_.rndm->debug_count_get), sizeof (brain_.rndm->debug_count_get));

        return true;
    }

    return false;
}

void brain_friend::resize(_word brainBits_)
{
    if(brain_.state_ != brain::state::state_stopped)
        throw "brain is running now";

    if(brainBits_ > brain_.quantity_of_neurons_in_power_of_two)
    {
        _word quantity_of_neuron_end_temp = 1 << (brainBits_);

        std::vector<bnn::brain::union_storage> us_temp = std::vector<brain::union_storage>(quantity_of_neuron_end_temp);

        for(_word i = 0; i < brain_.quantity_of_neurons; i++)
            for(_word j = 0; j < sizeof(brain::union_storage) / sizeof(_word); j++)
                us_temp[i].words[j] = brain_.us[i].words[j];

        for (_word i = brain_.quantity_of_neurons; i < quantity_of_neuron_end_temp; i++)
            us_temp[i].binary_ = brain::union_storage::binary();

        std::swap(brain_.us, us_temp);

        brain_.quantity_of_neurons_in_power_of_two = brainBits_;
        brain_.quantity_of_neurons = quantity_of_neuron_end_temp;
        brain_.quantity_of_neurons_binary = brain_.quantity_of_neurons - brain_.quantity_of_neurons_sensor - brain_.quantity_of_neurons_motor;
    }
}

bool brain_friend::save(std::ofstream& ofs)
{
    if(ofs.is_open())
    {
        ofs.write(reinterpret_cast<char*>(&brain_.quantity_of_neurons_in_power_of_two), sizeof (brain_.quantity_of_neurons_in_power_of_two));
        ofs.write(reinterpret_cast<char*>(&brain_.quantity_of_neurons_in_power_of_two_max), sizeof (brain_.quantity_of_neurons_in_power_of_two_max));
        ofs.write(reinterpret_cast<char*>(&brain_.quantity_of_neurons), sizeof (brain_.quantity_of_neurons));
        ofs.write(reinterpret_cast<char*>(&brain_.quantity_of_neurons_binary), sizeof (brain_.quantity_of_neurons_binary));
        ofs.write(reinterpret_cast<char*>(&brain_.quantity_of_neurons_sensor), sizeof (brain_.quantity_of_neurons_sensor));
        ofs.write(reinterpret_cast<char*>(&brain_.quantity_of_neurons_motor), sizeof (brain_.quantity_of_neurons_motor));
        ofs.write(reinterpret_cast<char*>(&brain_.iteration), sizeof (brain_.iteration));
        ofs.write(reinterpret_cast<char*>(&brain_.quantity_of_initialized_neurons_binary), sizeof (brain_.quantity_of_initialized_neurons_binary));

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
            for(_word j = 0; j < sizeof(brain::union_storage) / sizeof(_word); j++)
            {
                w = brain_.us[i].words[j];
                ofs.write(reinterpret_cast<char*>(&w), sizeof (w));
            }

        w = brain_.rndm->get_length();
        ofs.write(reinterpret_cast<char*>(&w), sizeof (w));

        for(_word i = 0; i < brain_.rndm->get_length(); i++)
        {
            w = brain_.rndm->get_array()[i];
            ofs.write(reinterpret_cast<char*>(&w), sizeof (w));
        }

        ofs.write(reinterpret_cast<char*>(&brain_.rndm->debug_count_put), sizeof (brain_.rndm->debug_count_put));
        ofs.write(reinterpret_cast<char*>(&brain_.rndm->debug_count_get), sizeof (brain_.rndm->debug_count_get));

        return true;
    }

    return false;
}

void brain_friend::stop()
{
    brain_.stop();
}

} // namespace bnn
