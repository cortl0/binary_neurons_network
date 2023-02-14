/*
 *   binary neurons network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef BNN_NEURONS_BINARY_IMPLEMENTATION_H
#define BNN_NEURONS_BINARY_IMPLEMENTATION_H

#include "neuron_implementation.h"

auto bnn_binary_set = [BNN_LAMBDA_REFERENCE](bnn_binary* me) -> void
{
    bnn_neuron_set(
            &me->neuron_,
            bnn_neuron::type::binary
            );

    me->in_work = false;
};

auto bnn_binary_calculate_body = [BNN_LAMBDA_REFERENCE](
        bnn_binary* me,
        const bnn_neuron* first,
        const bnn_neuron* second
        ) -> void
{
    static constexpr bool calculation_table[2][2][2][2] =
    {
        {
            {{1, 0}, {0, 0}},
            {{1, 0}, {1, 1}}
        },
        {
            {{0, 0}, {1, 0}},
            {{1, 1}, {1, 0}}
        }
    };

    me->neuron_.output_new = calculation_table
            [me->first_input_memory]
            [me->second_input_memory]
            [first->output_new]
            [second->output_new];
};

auto bnn_binary_init = [BNN_LAMBDA_REFERENCE](
        bnn_bnn* bnn,
        bnn_binary* me,
        bnn_neuron* first,
        bnn_neuron* second,
        u_word thread_number
        ) -> void
{
    me->first_input_memory = first->output_new;
    me->second_input_memory = second->output_new;
    bnn_binary_calculate_body(me, first, second);
    me->neuron_.output_old = me->neuron_.output_new;

    me->neuron_.level =
            first->level > second->level ?
                first->level + 1 : second->level + 1;

    me->first_input_life_counter = first->life_counter;
    me->second_input_life_counter = second->life_counter;
    me->in_work = true;

#ifdef DEBUG
    ++bnn->threads_.data[thread_number].debug_created;
#endif
};

auto bnn_binary_create = [BNN_LAMBDA_REFERENCE](
        bnn_bnn* bnn,
        bnn_binary* me,
        const u_word me_offset,
        u_word thread_number
        ) -> bool
{
    bnn_thread& thread = bnn->threads_.data[thread_number];

    me->first_input_address = thread.start_neuron +
                bnn_random_pull(&bnn->random_, thread.length_in_us_in_power_of_two, &thread.random_config);

    if(me_offset == me->first_input_address)
        return false;

    me->second_input_address = thread.start_neuron +
                bnn_random_pull(&bnn->random_, thread.length_in_us_in_power_of_two, &thread.random_config);

    if(me_offset == me->second_input_address)
        return false;

    if(me->first_input_address == me->second_input_address)
        return false;

    bnn_neuron* first = &bnn->storage_.data[me->first_input_address].neuron_;
    bnn_neuron* second = &bnn->storage_.data[me->second_input_address].neuron_;

    if(!((first->type_ == bnn_neuron::type::binary ? reinterpret_cast<bnn_binary*>(first)->in_work : false) ||
          (first->type_ == bnn_neuron::type::motor) ||
          (first->type_ == bnn_neuron::type::sensor)))
        return false;

    if(!((second->type_ == bnn_neuron::type::binary ? reinterpret_cast<bnn_binary*>(second)->in_work : false) ||
          (second->type_ == bnn_neuron::type::motor) ||
          (second->type_ == bnn_neuron::type::sensor)))
        return false;

    if((first->output_new == first->output_old) || (second->output_new == second->output_old))
        return false;

    bnn_binary_init(
            bnn,
            me,
            first,
            second,
            thread_number
            );

    ++bnn->threads_.data[thread_number].quantity_of_initialized_neurons_binary;

    return true;
};

auto bnn_binary_kill = [BNN_LAMBDA_REFERENCE](
        bnn_bnn* bnn,
        bnn_binary* me,
        u_word thread_number
        ) -> void
{
    me->in_work = false;
    me->neuron_.life_counter++;
    --bnn->threads_.data[thread_number].quantity_of_initialized_neurons_binary;

#ifdef DEBUG
    ++bnn->threads_.data[thread_number].debug_killed;
#endif
};

auto bnn_binary_calculate = [BNN_LAMBDA_REFERENCE](
        bnn_bnn* bnn,
        bnn_binary* me,
        const u_word me_offset,
        bnn_random_config* random_config,
        u_word thread_number
        ) -> void
{
    bnn_neuron_calculate(&me->neuron_);

    //bnn_binary* me = &bnn->storage_.data[me_offset].binary_;

    auto update_output_new_for_random_filling = [&]() -> void
    {
        me->neuron_.output_old = me->neuron_.output_new;
        me->neuron_.output_new = bnn_random_pull(&bnn->random_, 1, random_config);
    };

    bnn_neuron* first = &bnn->storage_.data[me->first_input_address].neuron_;
    bnn_neuron* second = &bnn->storage_.data[me->second_input_address].neuron_;

    if(me->in_work)
    {
        bool need_to_kill = false;

        if(bnn_neuron::type::binary == first->type_)
            if(first->life_counter != me->first_input_life_counter)
                need_to_kill = true;

        if(bnn_neuron::type::binary == second->type_)
            if(second->life_counter != me->second_input_life_counter)
                need_to_kill = true;

        if(need_to_kill)
        {
            bnn_binary_kill(bnn, me, thread_number);
            update_output_new_for_random_filling();
        }
        else
        {
            if((me_offset == bnn->parameters_.candidate_for_kill) &&
                (bnn->parameters_.quantity_of_initialized_neurons_binary * 3 > bnn->storage_.size * 2))
            {
                bnn_binary_kill(bnn, me, thread_number);
                update_output_new_for_random_filling();
            }
            else
            {
                me->neuron_.output_old = me->neuron_.output_new;
                bnn_binary_calculate_body(me, first, second);
            }
        }
    }
    else
    {
        if((bnn_random_pull_under(&bnn->random_, bnn->storage_.size - bnn->parameters_.quantity_of_initialized_neurons_binary, random_config)) ||
                (bnn->parameters_.quantity_of_initialized_neurons_binary * 3 < bnn->storage_.size * 2))
            if(!bnn_binary_create(bnn, me, me_offset, thread_number))
            {
                update_output_new_for_random_filling();
        }
    }

    bnn_neuron_push_random(&bnn->random_, &me->neuron_, random_config);
};

#endif // BNN_NEURONS_BINARY_IMPLEMENTATION_H
