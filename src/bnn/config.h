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
#define BNN_BYTES_ALIGNMENT sizeof(u_word)

enum bnn_error_codes
{
    ok,
    error = 2,
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

#ifdef DEBUG
struct debug
{
    struct consensus
    {
        u_word average{0};
        s_word max{0};
        u_word max_binary_num{~u_word(0)};
    };

    struct neuron
    {
        u_word calculation_count_min{~u_word(0)};
        u_word calculation_count_max{0};
    };

    struct random
    {
        unsigned long long int count_get{0};
        unsigned long long int count_put{0};
        long long int sum_put{0};
    };

    unsigned long long int created{0};
    unsigned long long int killed{0};
    consensus consensus_;
    u_word max_consensus_motor_num{~u_word(0)};
    neuron neuron_;
    random random_;
};
#endif

#endif // BNN_CONFIG_H
