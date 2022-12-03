/*
 *   binary neurons network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#include "motor.h"
#include "bnn.h"
#include "../common/headers/simple_math.hpp"
#include "storage.hpp"
#include "thread.h"

namespace bnn::neurons
{

motor::motor(const std::vector<bool>& world_output, u_word world_output_address)
    : world_output_address(world_output_address)
{
    type_ = neuron::type::motor;
    output_new = world_output[world_output_address];
    output_old = output_new;
}

void motor::solve(brain& b, const u_word thread_number, const u_word me)
{
    neuron::solve(b, thread_number, me);

#ifdef DEBUG
    s_word debug_average_consensus = 0;
    s_word debug_count = 0;
#endif

    for(auto it = binary_neurons.begin(); it != binary_neurons.end();)
    {
        if(it->second.life_counter != b.storage_[it->first]->life_counter)
        {
            it = binary_neurons.erase(it);
            continue;
        }

        accumulator += (b.storage_[it->first]->output_new * 2 - 1) * simple_math::sign0(it->second.consensus);

        if ((b.storage_[it->first]->output_new ^ b.storage_[it->first]->output_old) & (output_new ^ output_old))
            it->second.consensus -= ((b.storage_[it->first]->output_new ^ output_new) * 2 - 1);

#ifdef DEBUG
        if(b.threads[thread_number].debug_max_consensus < abs(it->second.consensus))
        {
            b.threads[thread_number].debug_max_consensus = abs(it->second.consensus);
            b.threads[thread_number].debug_max_consensus_binary_num = it->first;
            b.threads[thread_number].debug_max_consensus_motor_num = me;
        }

        debug_average_consensus += abs(it->second.consensus);
        debug_count++;
#endif
        it++;
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

    neuron::put_random(b, thread_number);
    b.world_output[world_output_address] = output_new;
    accumulator >>= 1;
    accumulator += (b.random_->get(1, b.threads[thread_number].random_config) << 1) - 1;

    //    if (brn.threads[thread_number].rndm->get_ft(0, binary_neurons->size()))
    //        return;

    //    if (brn.thrd[thread_number].rndm->get(binary_neurons->size()))
    //        return;

    u_word i = b.random_->get(b.quantity_of_neurons_in_power_of_two, b.threads[thread_number].random_config);

    if(b.storage_[i]->get_type() == neuron::neuron::type::binary && ((neurons::binary*)(b.storage_[i].get()))->in_work)
            //            if(std::none_of(binary_neurons->begin(), binary_neurons->end(), [&](const std::pair<_word, binary_neuron>& p)
            //            {
            //                return i == p.first;
            //            }))
        {
            if (((b.storage_[i]->output_new ^ b.storage_[i]->output_old)
                 & (output_new ^ output_old)))
                binary_neurons.insert(
                            std::pair<u_word, binary_neuron>(
                                i,
                                binary_neuron(
                                    i,
                                    b.storage_[i]->life_counter,
                                    0)));
        }
}

motor::binary_neuron::binary_neuron(u_word address, u_word life_counter, s_word consensus)
    : address(address), life_counter(life_counter), consensus(consensus)
{

}

} // namespace bnn::neurons
