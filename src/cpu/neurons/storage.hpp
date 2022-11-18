/*
 *   Binary Neurons Network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef BNN_STORAGE_HPP
#define BNN_STORAGE_HPP

#include "binary.h"
#include "motor.h"
#include "sensor.h"

namespace bnn
{

union storage final
{
    neurons::binary binary_;
    neurons::motor motor_;
    neurons::neuron neuron_;
    neurons::sensor sensor_;
    u_word words
    [
        (sizeof(neurons::motor) >= sizeof(neurons::binary) && sizeof(neurons::motor) >= sizeof(neurons::sensor) ?
             sizeof(neurons::motor) :
             sizeof(neurons::binary) >= sizeof(neurons::sensor) ?
                 sizeof(neurons::binary) : sizeof(neurons::sensor))
        / sizeof(u_word)
    ];
};

} // namespace bnn

#endif // BNN_STORAGE_HPP
