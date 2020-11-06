//*************************************************************//
//                                                             //
//   binary neurons network                                    //
//   created by Ilya Shishkin                                  //
//   cortl@8iter.ru                                            //
//   http://8iter.ru/ai.html                                   //
//   https://github.com/cortl0/binary_neurons_network          //
//   licensed by GPL v3.0                                      //
//                                                             //
//*************************************************************//

#include "../brain.h"

namespace bnn
{

brain::union_storage::motor::motor(std::vector<bool>& world_output, _word world_output_address_)
{
    neuron_type_ = neuron_type_motor;
    world_output_address = world_output_address_;
    out_new = world_output[world_output_address];
    out_old = out_new;
}

void brain::union_storage::motor::solve(brain &brn, _word me)
{
    out_old = out_new;
    if (accumulator < 0)
        out_new = false;
    else
        out_new = true;
    brn.world_output[world_output_address] = out_new;
    accumulator >>= 1;
    accumulator += (brn.rndm->get(1) << 1) - 1;
    if (brn.rndm->get_ft(0, slots_occupied))
        return;
    _word i = brn.rndm->get(brn.quantity_of_neurons_in_power_of_two);
    if(brn.us[i].neuron_.get_type()==brain::union_storage::neuron::neuron_type_binary)
        if(brn.us[i].binary_.get_type_binary()==brain::union_storage::binary::neuron_binary_type_in_work)
            if(!brn.us[i].binary_.motor_connect)
            {
                slots_occupied++;
                brn.us[i].binary_.motor_connect = true;
                brn.us[i].binary_.motor_consensus = 0;
                brn.us[i].binary_.motor = me;
            }
}

} // namespace bnn
