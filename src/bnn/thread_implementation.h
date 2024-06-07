/*
 *   Binary Neurons Network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef BNN_THREAD_IMPLEMENTATION_H
#define BNN_THREAD_IMPLEMENTATION_H

//#include "bnn_implementation.h"
#include "neurons/storage_implementation.h"

auto bnn_thread_function = [BNN_LAMBDA_REFERENCE](
        bnn_bnn* const bnn,
        u_word thread_number
        ) -> void
{
    bnn_thread* const me = &bnn->threads_.data[thread_number];

#ifdef DEBUG
    me->debug_.consensus_.average = 0;
    me->debug_.consensus_.max = 0;
    me->debug_.consensus_.max_binary_num = ~u_word(0);
    me->debug_.max_consensus_motor_num = ~u_word(0);
    me->debug_.neuron_.calculation_count_min = ~u_word(0);
    me->debug_.neuron_.calculation_count_max = 0;
#endif

    bnn_random_config* thread_random_config = &bnn->threads_.data[thread_number].random_config;
    u_word reaction_rate = bnn->threads_.neurons_per_thread;
    u_word current_neuron_offset;

    while(bnn->parameters_.state != bnn_state::started)
        ;

    me->in_work = true;

    while(bnn->parameters_.state == bnn_state::started)
    {
        if(!reaction_rate--)
        {
            reaction_rate = bnn->threads_.neurons_per_thread;
            ++me->iteration;

#ifdef BNN_ARCHITECTURE_CUDA
            if(thread_number == bnn->threads_.size - 1)
            {
                while(true)
                {
                    for(int t = 0; t < bnn->threads_.size - 1; ++t)
                        if(bnn->threads_.data[t].in_work)
                            continue;
                    break;
                }


                static u_word iteration_new;
                iteration_new = 0;
                static u_word quantity_of_initialized_neurons_binary;
                quantity_of_initialized_neurons_binary = 0;

                bnn->debug_.neuron_.calculation_count_max = 0;
                bnn->debug_.neuron_.calculation_count_min = ~u_word(0);
                bnn->debug_.created = 0;
                bnn->debug_.killed = 0;
                bnn->debug_.random_.count_get = 0;
                bnn->debug_.random_.count_put = 0;
                bnn->debug_.random_.sum_put = 0;

                bnn->debug_.consensus_.average = 0;
                bnn->debug_.consensus_.max = 0;
                bnn->debug_.consensus_.max_binary_num = ~u_word{0};
                bnn->debug_.max_consensus_motor_num = ~u_word{0};

                for(u_word i = 0; i < bnn->threads_.size; ++i)
                {
                    const auto& t = bnn->threads_.data[i];
                    iteration_new += t.iteration;
                    quantity_of_initialized_neurons_binary += t.quantity_of_initialized_neurons_binary;

                    if(bnn->debug_.neuron_.calculation_count_max < t.debug_.neuron_.calculation_count_max)
                        bnn->debug_.neuron_.calculation_count_max = t.debug_.neuron_.calculation_count_max;

                    if(bnn->debug_.neuron_.calculation_count_min > t.debug_.neuron_.calculation_count_min)
                        bnn->debug_.neuron_.calculation_count_min = t.debug_.neuron_.calculation_count_min;

                    bnn->debug_.created += t.debug_.created;
                    bnn->debug_.killed += t.debug_.killed;

                    bnn->debug_.random_.count_get += t.random_config.debug_.random_.count_get;
                    bnn->debug_.random_.count_put += t.random_config.debug_.random_.count_put;
                    bnn->debug_.random_.sum_put += t.random_config.debug_.random_.sum_put;

#ifdef DEBUG
                    bnn->debug_.consensus_.average += t.debug_.consensus_.average;

                    if(bnn->debug_.consensus_.max < t.debug_.consensus_.max)
                    {
                        bnn->debug_.consensus_.max = t.debug_.consensus_.max;
                        bnn->debug_.consensus_.max_binary_num = t.debug_.consensus_.max_binary_num;
                        bnn->debug_.max_consensus_motor_num = t.debug_.max_consensus_motor_num;
                    }
#endif
                }
#ifdef DEBUG
                bnn->debug_.consensus_.average /= bnn->threads_.size;
#endif

                bnn->parameters_.iteration = iteration_new / bnn->threads_.size;
                bnn->parameters_.quantity_of_initialized_neurons_binary = quantity_of_initialized_neurons_binary;

                bnn->parameters_.candidate_for_kill = bnn_random_pull(
                            &bnn->random_,
                            bnn->storage_.size_in_power_of_two,
                            &bnn->parameters_.random_config);
            }
#endif

#ifdef DEBUG
            //u_word debug_average_consensus = 0;
            u_word debug_count = 0;

            for(u_word i = me->start_neuron; i < me->start_neuron + bnn->threads_.neurons_per_thread; ++i)
                if(bnn_neuron::type::motor == bnn->storage_.data[i].neuron_.type_)
                {
                    if(me->debug_.consensus_.max < bnn->storage_.data[i].motor_.debug_max_consensus)
                        me->debug_.consensus_.max = bnn->storage_.data[i].motor_.debug_max_consensus;

                    me->debug_.consensus_.average += bnn->storage_.data[i].motor_.debug_average_consensus;
                    ++debug_count;
                }

            if(debug_count > 0)
                me->debug_.consensus_.average /= debug_count;

            if(me->debug_.consensus_.max > 0)
                --me->debug_.consensus_.max;
#endif

#ifdef BNN_ARCHITECTURE_CUDA
            break;
#endif
        }

        current_neuron_offset = me->start_neuron + bnn_random_pull(&bnn->random_, me->length_in_us_in_power_of_two, &me->random_config);
        bnn_storage* storage_item = &bnn->storage_.data[current_neuron_offset];

#ifdef DEBUG
        ++storage_item->neuron_.calculation_count;

        if(me->debug_.neuron_.calculation_count_max < storage_item->neuron_.calculation_count)
            me->debug_.neuron_.calculation_count_max = storage_item->neuron_.calculation_count;

        if(me->debug_.neuron_.calculation_count_min > storage_item->neuron_.calculation_count)
            me->debug_.neuron_.calculation_count_min = storage_item->neuron_.calculation_count;
#endif

        switch(bnn->storage_.data[current_neuron_offset].neuron_.type_)
        {
        case bnn_neuron::type::binary:
        {
            bnn_binary_calculate
                    (
                        bnn,
                        &storage_item->binary_,
                        current_neuron_offset,
                        thread_random_config,
                        thread_number
                    );
            break;
        }
        case bnn_neuron::type::sensor:
        {
            bnn_sensor_calculate
                    (
                        bnn,
                        &storage_item->sensor_,
                        thread_random_config
                    );
            break;
        }
        case bnn_neuron::type::motor:
        {
            bnn_motor_calculate
                    (
                        bnn,
                        &storage_item->motor_,
                        current_neuron_offset,
                        thread_random_config,
                        thread_number
                    );
            break;
        }
        default:
        {
            break;
        }
        }
    }

    me->in_work = false;
};

#endif // BNN_THREAD_IMPLEMENTATION_H
