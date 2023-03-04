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

auto bnn_binary_set = [BNN_LAMBDA_REFERENCE](
        bnn_binary* me,
        bool output_new,
        bool output_old
        ) -> void
{
    bnn_neuron_set(
            &me->neuron_,
            bnn_neuron::type::binary,
            output_new,
            output_old
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

    me->neuron_.output_old = me->neuron_.output_new;

    me->neuron_.output_new = calculation_table
            [me->first.memory]
            [me->second.memory]
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
    me->first.life_counter = first->life_counter;
    me->second.life_counter = second->life_counter;
    me->first.memory = first->output_new;
    me->second.memory = second->output_new;
    me->neuron_.output_old = me->neuron_.output_new;
    me->neuron_.level = first->level > second->level ? first->level + 1 : second->level + 1;
    me->in_work = true;
    ++bnn->threads_.data[thread_number].quantity_of_initialized_neurons_binary;

#ifdef DEBUG
    ++bnn->threads_.data[thread_number].debug_.created;
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

    me->first.address = bnn_random_pull(&bnn->random_, bnn->storage_.size_in_power_of_two, &thread.random_config);

    if(me_offset == me->first.address)
        return false;

    me->second.address = bnn_random_pull(&bnn->random_, bnn->storage_.size_in_power_of_two, &thread.random_config);

    if(me_offset == me->second.address)
        return false;

    if(me->first.address == me->second.address)
        return false;

    bnn_neuron* first = &bnn->storage_.data[me->first.address].neuron_;
    bnn_neuron* second = &bnn->storage_.data[me->second.address].neuron_;

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

    return true;
};

auto bnn_binary_create_primary_fake = [BNN_LAMBDA_REFERENCE](
        bnn_bnn* bnn,
        const u_word me_offset,
        const u_word input_offset,
        const u_word thread_number
        ) -> void
{
    bnn->storage_.data[me_offset].binary_.first.address = input_offset;
    bnn->storage_.data[me_offset].binary_.second.address = input_offset;

    bnn_neuron* first = &bnn->storage_.data[input_offset].neuron_;
    bnn_neuron* second = &bnn->storage_.data[input_offset].neuron_;

    bnn_binary_init(
            bnn,
            &bnn->storage_.data[me_offset].binary_,
            first,
            second,
            thread_number
            );
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
    ++bnn->threads_.data[thread_number].debug_.killed;
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
    if(!me->in_work)
    {
        bnn_binary_create(bnn, me, me_offset, thread_number);
        return;
    }

    bnn_neuron* first = &bnn->storage_.data[me->first.address].neuron_;
    bnn_neuron* second = &bnn->storage_.data[me->second.address].neuron_;

    bool need_to_kill = false;

    if(bnn_neuron::type::binary == first->type_)
        if(first->life_counter != me->first.life_counter)
            need_to_kill = true;

    if(bnn_neuron::type::binary == second->type_)
        if(second->life_counter != me->second.life_counter)
            need_to_kill = true;

    if((me_offset == bnn->parameters_.candidate_for_kill) &&
        (bnn->parameters_.quantity_of_initialized_neurons_binary > (bnn->storage_.size >> 1)))
        need_to_kill = true;

    if(need_to_kill)
    {
        bnn_binary_kill(bnn, me, thread_number);
        return;
    }

    bnn_binary_calculate_body(me, first, second);
    bnn_neuron_push_random(&bnn->random_, &me->neuron_, random_config);
};

#endif // BNN_NEURONS_BINARY_IMPLEMENTATION_H
