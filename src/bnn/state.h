/*
 *   Binary Neurons Network
 *   created by Ilya Shishkin
 *   cortl@yandex.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef BNN_STATE_H
#define BNN_STATE_H

enum class bnn_state : int
{
    start = 1,
    started,
    stop,
    stopped
};

#endif // BNN_STATE_H
