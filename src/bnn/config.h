/*
 *   Binary Neurons Network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef BNN_CONFIG_H
#define BNN_CONFIG_H

#include "types.h"

#ifndef DEBUG
#define DEBUG
#endif

#if not defined(BNN_ARCHITECTURE_CPU) && not defined(BNN_ARCHITECTURE_CUDA)
#error One architecture must be defined
#endif

#if defined(BNN_ARCHITECTURE_CPU) && defined(BNN_ARCHITECTURE_CUDA)
#error Only one architecture must be defined
#endif

#ifdef BNN_ARCHITECTURE_CPU
#define BNN_LAMBDA_REFERENCE
#elif defined BNN_ARCHITECTURE_CUDA
#define BNN_LAMBDA_REFERENCE &
#endif

#define QUANTITY_OF_BITS_IN_BYTE 8
#define QUANTITY_OF_BITS_IN_WORD (sizeof(u_word) * QUANTITY_OF_BITS_IN_BYTE)
#define BNN_BYTES_ALIGNMENT 8

enum bnn_error_codes
{
    ok,
    error,
    input_size_must_be_greater_than_zero,
    output_size_must_be_greater_than_zero,
    motor_binaries_size_per_motor_must_be_greater_than_zero,
    storage_size_too_small,
    neurons_per_thread_must_be_greater_than_zero,
    random_size_in_power_of_two_must_be_less_then_quantity_of_bits_in_word,

    invalid_value,
    out_of_range,
    malloc_fail
};

#endif // BNN_CONFIG_H
