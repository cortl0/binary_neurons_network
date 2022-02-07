/*
 *   Binary Neurons Network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef BNN_STORAGE_H
#define BNN_STORAGE_H

#include "neurons/neuron.h"
#include "neurons/binary.h"
#include "neurons/sensor.h"
#include "neurons/motor.h"

namespace bnn
{

union storage final
{
    neuron neuron_;
    binary binary_;
    sensor sensor_;
    motor motor_;
    _word words[sizeof(motor_) / sizeof(_word)];
    storage();
};

} // namespace bnn

#endif // BNN_STORAGE_H
