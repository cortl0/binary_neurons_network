/*
 *   Binary Neurons Network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef BNN_WORLD_H
#define BNN_WORLD_H

#include "config.h"

struct bnn_world
{
    bool* data{nullptr};
    u_word size{0};
};

#endif // BNN_WORLD_H
