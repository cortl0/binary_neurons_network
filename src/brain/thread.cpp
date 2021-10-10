/*
 *   binary neurons network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#include "thread.h"
#include "brain.h"
#include "neurons/neuron.h"
#include "storage.h"
#include "m_sequence.h"
#include "random/random.h"

namespace bnn
{

thread::thread(brain* brn,
                      _word thread_number,
                      _word start_neuron,
                      _word length_in_us_in_power_of_two,
                      random::config random_config)
    : thread_number(thread_number),
      length_in_us_in_power_of_two(length_in_us_in_power_of_two),
      start_neuron(start_neuron),
      random_config(random_config)
{
    thread_ = std::thread(function, brn, thread_number, start_neuron, length_in_us_in_power_of_two);
}

void thread::function(brain* brn, _word thread_number, _word start_in_us, _word length_in_us_in_power_of_two)
{
    _word reaction_rate = 0;

    _word j;

    _word quantity_of_neurons = simple_math::two_pow_x(brn->quantity_of_neurons_in_power_of_two) / brn->threads_count;

    do
    {
        sleep(1);
    }
    while(brn->threads[thread_number].in_work != true);

    while(brn->state_ != state::stop)
    {
        if(!reaction_rate--)
        {
            reaction_rate = quantity_of_neurons;

            brn->threads[thread_number].iteration++;

#ifdef DEBUG
            _word debug_average_consensus = 0;

            _word debug_count = 0;

            for(_word i = brn->threads[thread_number].start_neuron;
                i < brn->threads[thread_number].start_neuron + brn->quantity_of_neurons / brn->threads_count; i++)
                if(brn->storage_[i].neuron_.neuron_type_ == neuron::neuron_type::neuron_type_motor)
                {
                    debug_average_consensus += brn->storage_[i].motor_.debug_average_consensus;

                    debug_count++;
                }

            if(debug_count > 0)
                brn->threads[thread_number].debug_average_consensus = debug_average_consensus / debug_count;

            if(brn->threads[thread_number].debug_max_consensus > 0)
                brn->threads[thread_number].debug_max_consensus--;
#endif
        }

        j = start_in_us + brn->random_->get(length_in_us_in_power_of_two, brn->threads[thread_number].random_config);

        brn->storage_[j].neuron_.solve(*brn, j, thread_number);
    }

    brn->threads[thread_number].in_work = false;
}

} // namespace bnn
