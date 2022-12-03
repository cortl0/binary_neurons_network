/*
 *   binary neurons network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#include "neuron.h"
#include "bnn.h"
#include "storage.hpp"
#include "thread.h"

namespace bnn::neurons
{

neuron::neuron()
{

}

const neuron::type& neuron::get_type() const
{
    return type_;
}

void neuron::put_random(brain& b, const u_word thread_number)
{
    if (output_new != output_old)
        b.random_->put(output_new, b.threads[thread_number].random_config);
}

void neuron::solve(brain&, const u_word, const u_word)
{
#ifdef DEBUG
    calculation_count++;
#endif
}

} // namespace bnn::neurons
