/*
 *   binary neurons network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef STORAGE_H
#define STORAGE_H

#include "neurons/neuron.h"
#include "neurons/binary.h"
#include "neurons/sensor.h"
#include "neurons/motor.h"

namespace bnn
{

union storage
{
    neuron neuron_;
    binary binary_;
    sensor sensor_;
    motor motor_;
    _word words[sizeof(binary) / sizeof(_word)];
    storage(){}
};

} // namespace bnn

#endif // STORAGE_H
