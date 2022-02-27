/*
 *   Binary Neurons Network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#include "thread.h"

#include <unistd.h>

#include <iostream>

#include "brain.h"
#include "neurons/neuron.h"
#include "storage.hpp"
#include "m_sequence.h"
#include "random/random.h"

namespace bnn
{

thread::thread()
{

}

thread::thread(brain* brain_,
               u_word thread_number,
               u_word start_neuron,
               u_word length_in_us_in_power_of_two,
               random::config& random_config)
    : random_config(random_config),
      length_in_us_in_power_of_two(length_in_us_in_power_of_two),
      start_neuron(start_neuron),
      thread_number(thread_number),
      brain_(brain_)
{

}

void thread::start()
{
    logging("thread[" + std::to_string(thread_number) + "]::start() begin");

    thread_ = std::thread(function, this, brain_, start_neuron, length_in_us_in_power_of_two);

    thread_.detach();

    state_ = state::start;

    logging("thread[" + std::to_string(thread_number) + "]::start() end");
}

void thread::function(thread* me, brain* b, const u_word start_in_us, const u_word length_in_us_in_power_of_two)
{
    try
    {
        u_word reaction_rate = 0;

        u_word j;

        u_word quantity_of_neurons = b->quantity_of_neurons / b->threads_count;

        while(state::start != me->state_);

        me->state_ = state::started;

        logging("thread [" + std::to_string(me->thread_number) + "] started");

        while(state::started == me->state_)
        {
            if(!reaction_rate--)
            {
                reaction_rate = quantity_of_neurons;

                me->iteration++;

#ifdef DEBUG
                u_word debug_average_consensus = 0;

                u_word debug_count = 0;

                for(u_word i = b->threads[me->thread_number].start_neuron;
                    i < b->threads[me->thread_number].start_neuron + b->quantity_of_neurons / b->threads_count; i++)
                    if(b->storage_[i].neuron_.type_ == neurons::neuron::type::motor)
                    {
                        debug_average_consensus += b->storage_[i].motor_.debug_average_consensus;

                        debug_count++;
                    }

                if(debug_count > 0)
                    b->threads[me->thread_number].debug_average_consensus = debug_average_consensus / debug_count;

                if(b->threads[me->thread_number].debug_max_consensus > 0)
                    b->threads[me->thread_number].debug_max_consensus--;
#endif
            }

            j = start_in_us + b->random_->get(length_in_us_in_power_of_two, me->random_config);

            b->storage_[j].neuron_.solve(*b, j, me->thread_number);
        }
    }
    catch (...)
    {
        logging("error in thread [" + std::to_string(me->thread_number) + "]");
    }

    me->state_ = state::stopped;

    logging("thread [" + std::to_string(me->thread_number) + "] stopped");
}

} // namespace bnn
