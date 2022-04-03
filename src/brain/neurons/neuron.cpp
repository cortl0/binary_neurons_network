/*
 *   binary neurons network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#include "neuron.h"
#include "../brain.h"
#include "../storage.hpp"

namespace bnn::neurons
{

neuron::neuron()
{

}

const neuron::type& neuron::get_type() const
{
    return type_;
}

void neuron::solve(brain&, const u_word, const u_word)
{
#ifdef DEBUG
    calculation_count++;
#endif
}

} // namespace bnn::neurons
