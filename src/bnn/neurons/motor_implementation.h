/*
 *   binary neurons network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef BNN_NEURONS_MOTOR_IMPLEMENTATION_H
#define BNN_NEURONS_MOTOR_IMPLEMENTATION_H

#include "neuron_implementation.h"

auto bnn_motor_set = [BNN_LAMBDA_REFERENCE](
        bnn_bnn* bnn,
        bnn_motor* me,
        u_word world_output_address
        ) -> void
{
    bnn_neuron_set(
            &me->neuron_,
            bnn_neuron::type::motor,
            bnn->output_.data[world_output_address],
            bnn->output_.data[world_output_address]
            );

    me->world_output_address = world_output_address;
    me->accumulator = 0;

#ifdef DEBUG
    me->debug_average_consensus = 0;
#endif
};

auto bnn_motor_calculate = [BNN_LAMBDA_REFERENCE](
        bnn_bnn* bnn,
        bnn_motor* me,
        u_word me_offset,
        bnn_random_config* random_config,
        u_word thread_number
        ) ->void
{
    bnn_neuron_calculate(&me->neuron_);

#ifdef DEBUG
    s_word debug_average_consensus = 0;
    s_word debug_count = 0;
#endif

    u_word ii;

    for(u_word i = 0; i < bnn->motor_binaries_.size_per_motor; ++i)
    {
        ii = me->world_output_address * bnn->motor_binaries_.size_per_motor + i;

        if(bnn->motor_binaries_.data[ii].present)
            if(bnn->motor_binaries_.data[ii].life_counter !=
                    bnn->storage_.data[bnn->motor_binaries_.data[ii].address].neuron_.life_counter)
                bnn->motor_binaries_.data[ii].present = false;

        if(bnn->motor_binaries_.data[ii].present)
            me->accumulator += (bnn->storage_.data[bnn->motor_binaries_.data[ii].address].neuron_.output_new * 2 - 1) *
                bnn_math_sign0(bnn->motor_binaries_.data[ii].consensus);

        if(bnn->motor_binaries_.data[ii].present &&
                (bnn->storage_.data[bnn->motor_binaries_.data[ii].address].neuron_.output_new ^
                 bnn->storage_.data[bnn->motor_binaries_.data[ii].address].neuron_.output_old) &
                (me->neuron_.output_new ^ me->neuron_.output_old))
            bnn->motor_binaries_.data[ii].consensus -=
                    ((bnn->storage_.data[bnn->motor_binaries_.data[ii].address].neuron_.output_new ^
                     me->neuron_.output_new) * 2 - 1);

#ifdef DEBUG
        if(bnn->threads_.data[thread_number].debug_max_consensus < bnn_math_abs(bnn->motor_binaries_.data[ii].consensus))
        {
            bnn->threads_.data[thread_number].debug_max_consensus = bnn_math_abs(bnn->motor_binaries_.data[ii].consensus);
            bnn->threads_.data[thread_number].debug_max_consensus_binary_num = bnn->motor_binaries_.data[ii].address;
            bnn->threads_.data[thread_number].debug_max_consensus_motor_num = me_offset;
        }

        bnn->threads_.data[thread_number].debug_average_consensus += bnn_math_abs(bnn->motor_binaries_.data[ii].consensus);
        debug_count++;
#endif
    }

#ifdef DEBUG
    if(debug_count > 0)
        me->debug_average_consensus = debug_average_consensus / debug_count;
#endif

    me->neuron_.output_old = me->neuron_.output_new;
    if(me->accumulator < 0)
        me->neuron_.output_new = false;
    else if(me->accumulator > 0)
        me->neuron_.output_new = true;

    bnn_neuron_push_random(&bnn->random_, &me->neuron_, random_config);
    bnn->output_.data[me->world_output_address] = me->neuron_.output_new;
    me->accumulator >>= 1;
    me->accumulator += (bnn_random_pull(&bnn->random_, 1, random_config) << 1) - 1;

    //    if (brn.threads[thread_number].rndm->get_ft(0, binary_neurons->size()))
    //        return;

    //    if (brn.thrd[thread_number].rndm->get(binary_neurons->size()))
    //        return;

    u_word k = bnn_random_pull(&bnn->random_, bnn->storage_.size_in_power_of_two, random_config);

    if(bnn_neuron::type::binary == bnn->storage_.data[k].neuron_.type_ && bnn->storage_.data[k].binary_.in_work)
            //            if(std::none_of(binary_neurons->begin(), binary_neurons->end(), [&](const std::pair<_word, binary_neuron>& p)
            //            {
            //                return i == p.first;
            //            }))
    {
        if(((bnn->storage_.data[k].neuron_.output_new ^ bnn->storage_.data[k].neuron_.output_old)
             & (me->neuron_.output_new ^ me->neuron_.output_old)))
        {
            for(u_word i = 0; i < bnn->motor_binaries_.size_per_motor; ++i)
            {
                ii = me->world_output_address * bnn->motor_binaries_.size_per_motor + i;

                if(!bnn->motor_binaries_.data[ii].present)
                {
                    bnn->motor_binaries_.data[ii].address = k;
                    bnn->motor_binaries_.data[ii].life_counter = bnn->storage_.data[k].neuron_.life_counter;
                    bnn->motor_binaries_.data[ii].consensus = 0;
                    bnn->motor_binaries_.data[ii].present = true;
                    break;
                }
            }
        }
    }
};

#endif // BNN_NEURONS_MOTOR_IMPLEMENTATION_H
