/*
 *   Binary Neurons Network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef BNN_NEURONS_STORAGE_H
#define BNN_NEURONS_STORAGE_H

#include "neuron.h"
#include "sensor.h"
#include "binary.h"
#include "motor.h"

union bnn_storage
{
    bnn_binary binary_;
    bnn_motor motor_;
    bnn_neuron neuron_;
    bnn_sensor sensor_;
    u_word words
    [
        (sizeof(bnn_motor) >= sizeof(bnn_binary) && sizeof(bnn_motor) >= sizeof(bnn_sensor) ?
             sizeof(bnn_motor) :
             sizeof(bnn_binary) >= sizeof(bnn_sensor) ?
                 sizeof(bnn_binary) : sizeof(bnn_sensor))
        / sizeof(u_word)
    ];
};

struct bnn_storage_array
{
    bnn_storage* data{nullptr};
    u_word size{0};
    u_word size_in_power_of_two{0};
};

#endif // BNN_NEURONS_STORAGE_H
