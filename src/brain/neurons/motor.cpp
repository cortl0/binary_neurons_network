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
#include "../storage.h"

namespace bnn
{

motor::motor(std::vector<bool>& world_output, _word world_output_address_)
{
    neuron_type_ = neuron_type_motor;
    world_output_address = world_output_address_;
    out_new = world_output[world_output_address];
    out_old = out_new;
    binary_neurons = new std::map<_word, binary_neuron>();
}

void motor::solve(brain &brn, const _word &me, const _word &thread_number)
{
#ifdef DEBUG
    _word debug_average_consensus = 0;
    _word debug_count = 0;
#endif

    for(auto i = binary_neurons->begin(); i != binary_neurons->end();)
    {
        if(i->second.life_number != brn.storage_[i->first].binary_.life_number)
        {
            i = binary_neurons->erase(i);
            continue;
        }

        accumulator += (brn.storage_[i->first].binary_.out_new * 2 - 1) * simple_math::sign0(i->second.consensus);

        //        if ((brn.storage_[i->first].binary_.out_new ^ brn.storage_[i->first].binary_.out_old)
        //                & (out_new ^ out_old))
        if ((brn.storage_[i->first].binary_.out_new ^ brn.storage_[i->first].binary_.out_old)
                & (out_new ^ out_old))
            i->second.consensus -= ((brn.storage_[i->first].binary_.out_new ^ out_new) * 2 - 1);

#ifdef DEBUG
        if(brn.threads[thread_number].debug_max_consensus < abs(i->second.consensus))
        {
            brn.threads[thread_number].debug_max_consensus = abs(i->second.consensus);
            brn.threads[thread_number].debug_max_consensus_binary_num = i->first;
            brn.threads[thread_number].debug_max_consensus_motor_num = me;
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

    out_old = out_new;
    if (accumulator < 0)
        out_new = false;
    else
        out_new = true;
    brn.world_output[world_output_address] = out_new;
    accumulator >>= 1;
    accumulator += (brn.threads[thread_number].rndm->get(1) << 1) - 1;

    //    if (brn.threads[thread_number].rndm->get_ft(0, binary_neurons->size()))
    //        return;

    //    if (brn.thrd[thread_number].rndm->get(binary_neurons->size()))
    //        return;


    _word i = brn.threads[thread_number].rndm->get(brn.quantity_of_neurons_in_power_of_two);
    if(brn.storage_[i].neuron_.get_type()==neuron::neuron_type_binary)
        if(brn.storage_[i].binary_.get_type_binary()==binary::neuron_binary_type_in_work)
            //            if(std::none_of(binary_neurons->begin(), binary_neurons->end(), [&](const std::pair<_word, binary_neuron>& p)
            //            {
            //                return i == p.first;
            //            }))
        {
            if (((brn.storage_[i].binary_.out_new ^ brn.storage_[i].binary_.out_old)
                 & (out_new ^ out_old)))
                binary_neurons->insert(std::pair<_word, binary_neuron>(i, binary_neuron(i, brn.storage_[i].binary_.life_number)));
        }

    //std::cout << "binary_neurons->size() " << std::to_string(binary_neurons->size()) << std::endl;


    //std::cout << "accumulator " << std::to_string(accumulator) << std::endl;

}

motor::binary_neuron::binary_neuron(_word adress, _word life_number)
    : adress(adress), life_number(life_number)
{

}

} // namespace bnn
