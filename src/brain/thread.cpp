/*
 *   binary neurons network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#include "brain.h"

namespace bnn
{

brain::thread::thread(brain* brn,
                      _word thread_number,
                      _word start_neuron,
                      _word length_in_us_in_power_of_two,
                      _word random_array_length_in_power_of_two,
                      m_sequence& m_sequence)
    : thread_number(thread_number),
      start_neuron(start_neuron),
      length_in_us_in_power_of_two(length_in_us_in_power_of_two),
      random_array_length_in_power_of_two(random_array_length_in_power_of_two)
{
    rndm.reset(new random_put_get(random_array_length_in_power_of_two, m_sequence));
    thread_ = std::thread(function, brn, thread_number, start_neuron, length_in_us_in_power_of_two);
}

void brain::thread::function(brain* brn, _word thread_number, _word start_in_us, _word length_in_us_in_power_of_two)
{
    _word reaction_rate = 0;

    _word j;

    _word quantity_of_neurons = simple_math::two_pow_x(brn->quantity_of_neurons_in_power_of_two) / brn->threads_count;

    while(brn->threads[thread_number].in_work != true)
        ;

    while(brn->state_ != state::state_to_stop)
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
                if(brn->us[i].neuron_.neuron_type_ == union_storage::neuron::neuron_type::neuron_type_motor)
                {
                    debug_average_consensus += brn->us[i].motor_.debug_average_consensus;

                    debug_count++;
                }

            if(debug_count > 0)
                brn->threads[thread_number].debug_average_consensus = debug_average_consensus / debug_count;

            if(brn->threads[thread_number].debug_max_consensus > 0)
                brn->threads[thread_number].debug_max_consensus--;
#endif
        }

        j = start_in_us + brn->threads[thread_number].rndm->get(length_in_us_in_power_of_two);

        brn->us[j].neuron_.solve(*brn, j, thread_number);
    }

    brn->threads[thread_number].in_work = false;
}

} // namespace bnn
