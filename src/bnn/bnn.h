/*
 *   Binary Neurons Network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef BNN_BNN_H
#define BNN_BNN_H

#include "neurons/storage.h"
#include "parameters.h"
#include "thread.h"
#include "world.h"

struct bnn_bnn
{
    struct memory
    {
        void* data{nullptr};
        u_word size{0};
    } memory_;

    bnn_parameters parameters_;
    bnn_world input_;
    bnn_world output_;
    bnn_random random_;
    bnn_storage_array storage_;
    bnn_motor::binaries motor_binaries_;
    bnn_threads threads_;
};

#endif // BNN_BNN_H
