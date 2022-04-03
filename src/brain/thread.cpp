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
    if(in_work)
        return;

    thread_.reset(new std::thread(function, this, brain_));
    thread_->detach();
}

void thread::function(thread* me, brain* b)
{
    try
    {
        me->in_work = true;
        u_word reaction_rate = 0;
        u_word j;
        u_word quantity_of_neurons = b->quantity_of_neurons / b->threads_count;
        logging("thread [" + std::to_string(me->thread_number) + "] started");

        while(b->treads_to_work)
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
                    if(b->storage_[i]->get_type() == neurons::neuron::type::motor)
                    {
                        debug_average_consensus += ((neurons::motor*)(b->storage_[i].get()))->debug_average_consensus;
                        debug_count++;
                    }

                if(debug_count > 0)
                    b->threads[me->thread_number].debug_average_consensus = debug_average_consensus / debug_count;

                if(b->threads[me->thread_number].debug_max_consensus > 0)
                    b->threads[me->thread_number].debug_max_consensus--;
#endif
            }

            j = me->start_neuron + b->random_->get(me->length_in_us_in_power_of_two, me->random_config);
            b->storage_[j]->solve(*b, me->thread_number, j);
        }
    }
    catch (...)
    {
        logging("error in thread [" + std::to_string(me->thread_number) + "]");
    }

    logging("thread [" + std::to_string(me->thread_number) + "] stopped");
    me->in_work = false;
}

} // namespace bnn
