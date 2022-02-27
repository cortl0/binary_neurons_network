/*
 *   binary neurons network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#include "motor.h"
#include "../brain.h"
#include "../thread.h"
#include "../storage.hpp"

namespace bnn::neurons
{

motor::motor(const std::vector<bool>& world_output, u_word world_output_address)
    : world_output_address(world_output_address)
{
    type_ = neuron::type::motor;
    output_new = world_output[world_output_address];
    output_old = output_new;
    binary_neurons = new std::map<u_word, binary_neuron>();
}

void motor::solve(brain& b, const u_word me, const u_word thread_number)
{
#ifdef DEBUG
    s_word debug_average_consensus = 0;
    s_word debug_count = 0;
#endif

    for(auto i = binary_neurons->begin(); i != binary_neurons->end();)
    {
        if(i->second.life_counter != b.storage_[i->first].binary_.life_counter)
        {
            i = binary_neurons->erase(i);
            continue;
        }

        accumulator += (b.storage_[i->first].binary_.output_new * 2 - 1) * simple_math::sign0(i->second.consensus);

        //        if ((brn.storage_[i->first].binary_.out_new ^ brn.storage_[i->first].binary_.out_old)
        //                & (out_new ^ out_old))
        if ((b.storage_[i->first].binary_.output_new ^ b.storage_[i->first].binary_.output_old)
                & (output_new ^ output_old))
            i->second.consensus -= ((b.storage_[i->first].binary_.output_new ^ output_new) * 2 - 1);

#ifdef DEBUG
        if(b.threads[thread_number].debug_max_consensus < abs(i->second.consensus))
        {
            b.threads[thread_number].debug_max_consensus = abs(i->second.consensus);
            b.threads[thread_number].debug_max_consensus_binary_num = i->first;
            b.threads[thread_number].debug_max_consensus_motor_num = me;
        }

        debug_average_consensus += abs(i->second.consensus);
        debug_count++;
#endif
        i++;
    }

#ifdef DEBUG
    if(debug_count > 0)
        this->debug_average_consensus = debug_average_consensus / debug_count;
#endif

    output_old = output_new;
    if (accumulator < 0)
        output_new = false;
    else
        output_new = true;
    b.world_output[world_output_address] = output_new;
    accumulator >>= 1;
    accumulator += (b.random_->get(1, b.threads[thread_number].random_config) << 1) - 1;

    //    if (brn.threads[thread_number].rndm->get_ft(0, binary_neurons->size()))
    //        return;

    //    if (brn.thrd[thread_number].rndm->get(binary_neurons->size()))
    //        return;


    u_word i = b.random_->get(b.quantity_of_neurons_in_power_of_two, b.threads[thread_number].random_config);
    if(b.storage_[i].neuron_.get_type() == neuron::neuron::type::binary && b.storage_[i].binary_.in_work)
            //            if(std::none_of(binary_neurons->begin(), binary_neurons->end(), [&](const std::pair<_word, binary_neuron>& p)
            //            {
            //                return i == p.first;
            //            }))
        {
            if (((b.storage_[i].binary_.output_new ^ b.storage_[i].binary_.output_old)
                 & (output_new ^ output_old)))
                binary_neurons->insert(
                            std::pair<u_word, binary_neuron>(
                                i,
                                binary_neuron(
                                    i,
                                    b.storage_[i].binary_.life_counter,
                                    0)));
        }

    //std::cout << "binary_neurons->size() " << std::to_string(binary_neurons->size()) << std::endl;


    //std::cout << "accumulator " << std::to_string(accumulator) << std::endl;

}

motor::binary_neuron::binary_neuron(u_word address, u_word life_counter, s_word consensus)
    : address(address), life_counter(life_counter), consensus(consensus)
{

}

} // namespace bnn::neurons
