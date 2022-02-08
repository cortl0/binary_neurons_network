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
#include "storage.h"
#include "m_sequence.h"
#include "random/random.h"

namespace bnn
{

thread::thread()
{

}

thread::thread(brain* brain_,
               _word thread_number,
               _word start_neuron,
               _word length_in_us_in_power_of_two,
               random::config &random_config)
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

void thread::function(thread* me, brain* brain_, _word start_in_us, _word length_in_us_in_power_of_two)
{
    try
    {
        _word reaction_rate = 0;

        _word j;

        _word quantity_of_neurons = brain_->quantity_of_neurons / brain_->threads_count;

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
                _word debug_average_consensus = 0;

                _word debug_count = 0;

                for(_word i = brain_->threads[me->thread_number].start_neuron;
                    i < brain_->threads[me->thread_number].start_neuron + brain_->quantity_of_neurons / brain_->threads_count; i++)
                    if(brain_->storage_[i].neuron_.type_ == neuron::type::motor)
                    {
                        debug_average_consensus += brain_->storage_[i].motor_.debug_average_consensus;

                        debug_count++;
                    }

                if(debug_count > 0)
                    brain_->threads[me->thread_number].debug_average_consensus = debug_average_consensus / debug_count;

                if(brain_->threads[me->thread_number].debug_max_consensus > 0)
                    brain_->threads[me->thread_number].debug_max_consensus--;
#endif
            }

            j = start_in_us + brain_->random_->get(length_in_us_in_power_of_two, me->random_config);

            brain_->storage_[j].neuron_.solve(*brain_, j, me->thread_number);
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
