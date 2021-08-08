/*
 *   binary neurons network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#include "../brain.h"

namespace bnn
{

brain::union_storage::motor::motor(std::vector<bool>& world_output, _word world_output_address_)
{
    neuron_type_ = neuron_type_motor;
    world_output_address = world_output_address_;
    out_new = world_output[world_output_address];
    out_old = out_new;
    binary_neurons = new std::map<_word, binary_neuron>();
}

void brain::union_storage::motor::solve(brain &brn, _word me, _word thread_number)
{
#ifdef DEBUG
    _word debug_average_consensus = 0;
    _word count = 0;
#endif

    for(auto i = binary_neurons->begin(); i != binary_neurons->end();)
    {
        if(i->second.life_number != brn.us[i->first].binary_.life_number)
        {
            i = binary_neurons->erase(i);
            continue;
        }

        accumulator += (brn.us[i->first].binary_.out_new * 2 - 1) * simple_math::sign0(i->second.consensus);

        //        if ((brn.us[i->first].binary_.out_new ^ brn.us[i->first].binary_.out_old)
        //                & (out_new ^ out_old))
        if ((brn.us[i->first].binary_.out_new ^ brn.us[i->first].binary_.out_old)
                & (out_new ^ out_old))
            i->second.consensus -= ((brn.us[i->first].binary_.out_new ^ out_new) * 2 - 1);

#ifdef DEBUG
        if(brn.threads[thread_number].debug_max_consensus < abs(i->second.consensus))
            brn.threads[thread_number].debug_max_consensus = abs(i->second.consensus);

        debug_average_consensus += abs(i->second.consensus);
        count++;
#endif
        i++;
    }

    if(count > 0)
    this->debug_average_consensus = debug_average_consensus / count;

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
    if(brn.us[i].neuron_.get_type()==brain::union_storage::neuron::neuron_type_binary)
        if(brn.us[i].binary_.get_type_binary()==brain::union_storage::binary::neuron_binary_type_in_work)
            //            if(std::none_of(binary_neurons->begin(), binary_neurons->end(), [&](const std::pair<_word, binary_neuron>& p)
            //            {
            //                return i == p.first;
            //            }))
        {
            if (((brn.us[i].binary_.out_new ^ brn.us[i].binary_.out_old)
                 & (out_new ^ out_old)))
                binary_neurons->insert(std::pair<_word, binary_neuron>(i, binary_neuron(i, brn.us[i].binary_.life_number)));
        }

    //std::cout << "binary_neurons->size() " << std::to_string(binary_neurons->size()) << std::endl;


    //std::cout << "accumulator " << std::to_string(accumulator) << std::endl;

}

brain::union_storage::motor::binary_neuron::binary_neuron(_word adress, _word life_number)
    : adress(adress), life_number(life_number)
{

}

} // namespace bnn
