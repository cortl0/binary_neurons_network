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
#include "neurons/storage.hpp"
#include "../common/headers/m_sequence.h"

namespace bnn
{

thread::thread()
{
    logging("");
}

thread::thread(brain* brain_,
               u_word thread_number,
               u_word start_neuron,
               u_word length_in_us_in_power_of_two,
               random::random::config& random_config)
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

    std::thread(&thread::function, this).detach();
}

void thread::function()
{
    try
    {
        in_work = true;
        u_word reaction_rate = 0;
        u_word j;
        u_word quantity_of_neurons = brain_->quantity_of_neurons / brain_->threads.size();
        logging("thread [" + std::to_string(thread_number) + "] started");

        while(brain_->treads_to_work)
        {
            if(!reaction_rate--)
            {
                reaction_rate = quantity_of_neurons;
                iteration++;

#ifdef DEBUG
                u_word debug_average_consensus = 0;
                u_word debug_count = 0;

                for(u_word i = brain_->threads[thread_number].start_neuron;
                    i < brain_->threads[thread_number].start_neuron + brain_->quantity_of_neurons / brain_->threads.size(); i++)
                    if(brain_->storage_[i]->get_type() == neurons::neuron::type::motor)
                    {
                        debug_average_consensus += ((neurons::motor*)(brain_->storage_[i].get()))->debug_average_consensus;
                        debug_count++;
                    }

                if(debug_count > 0)
                    brain_->threads[thread_number].debug_average_consensus = debug_average_consensus / debug_count;

                if(brain_->threads[thread_number].debug_max_consensus > 0)
                    brain_->threads[thread_number].debug_max_consensus--;
#endif
            }

            j = start_neuron + brain_->random_->get(length_in_us_in_power_of_two, random_config);
            brain_->storage_[j]->solve(*brain_, thread_number, j);
        }
    }
    catch (...)
    {
        logging("error in thread [" + std::to_string(thread_number) + "]");
    }

    logging("thread [" + std::to_string(thread_number) + "] stopped");
    in_work = false;
}

} // namespace bnn
