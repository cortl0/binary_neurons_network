/*
 *   binary neurons network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef BNN_NEURONS_SENSOR_H
#define BNN_NEURONS_SENSOR_H

#include "neuron.h"

struct bnn_sensor
{
    bnn_neuron neuron_;
    u_word world_input_address{~u_word{0}};
};

#endif // BNN_NEURONS_SENSOR_H
