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
        bnn_bnn* bnn,
        u_word thread_number
        ) -> void
{
    bnn_thread* me = &bnn->threads_.data[thread_number];

    me->in_work = true;
    u_word reaction_rate = bnn->threads_.neurons_per_thread;
    u_word j;

    while(!bnn->parameters_.start)
        ;

    while(!bnn->parameters_.stop)
    {
        if(!reaction_rate--)
        {
            reaction_rate = bnn->threads_.neurons_per_thread;
            ++me->iteration;

#ifdef BNN_ARCHITECTURE_CUDA
            if(!thread_number)
            {
                static u_word iteration_new;
                iteration_new = 0;
                static u_word quantity_of_initialized_neurons_binary;
                quantity_of_initialized_neurons_binary = 0;

                for(u_word i = 0; i < bnn->threads_.size; ++i)
                {
                    iteration_new += bnn->threads_.data[i].iteration;
                    quantity_of_initialized_neurons_binary += bnn->threads_.data[i].quantity_of_initialized_neurons_binary;
                }

                bnn->parameters_.iteration = iteration_new / bnn->threads_.size;
                bnn->parameters_.quantity_of_initialized_neurons_binary = quantity_of_initialized_neurons_binary;

                bnn->parameters_.candidate_for_kill = bnn_random_pull(
                            &bnn->random_,
                            bnn->storage_.size_in_power_of_two,
                            &bnn->parameters_.random_config);
            }
#endif

#ifdef DEBUG
            u_word debug_average_consensus = 0;
            u_word debug_count = 0;

            for(u_word i = me->start_neuron; i < me->start_neuron + bnn->threads_.neurons_per_thread; ++i)
                if(bnn_neuron::type::motor == bnn->storage_.data[i].neuron_.type_)
                {
                    me->debug_average_consensus += bnn->storage_.data[i].motor_.debug_average_consensus;
                    debug_count++;
                }

            if(debug_count > 0)
                me->debug_average_consensus = debug_average_consensus / debug_count;

            if(me->debug_max_consensus > 0)
                me->debug_max_consensus--;
#endif

#ifdef BNN_ARCHITECTURE_CUDA
            break;
#endif
        }

        j = me->start_neuron + bnn_random_pull(&bnn->random_, me->length_in_us_in_power_of_two, &me->random_config);
        bnn_neuron_calculate(&bnn->storage_.data[j].neuron_);

        switch(bnn->storage_.data[j].neuron_.type_)
        {
        case bnn_neuron::type::binary:
        {
            bnn_binary_calculate
                    (
                        bnn,
                        &bnn->storage_.data[j].binary_,
                        j,
                        &bnn->threads_.data[thread_number].random_config,
                        thread_number
                    );
            break;
        }
        case bnn_neuron::type::sensor:
        {
            bnn_sensor_calculate
                    (
                        &bnn->storage_.data[j].sensor_,
                        &bnn->input_,
                        &bnn->random_,
                        &bnn->threads_.data[thread_number].random_config
                    );
            break;
        }
        case bnn_neuron::type::motor:
        {
            bnn_motor_calculate
                    (
                        bnn,
                        &bnn->storage_.data[j].motor_,
                        j,
                        &bnn->threads_.data[thread_number].random_config,
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

    //logging("thread [" + std::to_string(thread_number) + "] stopped");
    me->in_work = false;
};

#endif // BNN_THREAD_IMPLEMENTATION_H