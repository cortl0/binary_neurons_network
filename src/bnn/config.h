/*
 *   Binary Neurons Network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef CONFIG_H
#define CONFIG_H

#ifndef DEBUG
#define DEBUG
#endif

#define BNN_BYTES_ALIGNMENT 8

#ifdef BNN_ARCHITECTURE_CPU
#ifdef BNN_ARCHITECTURE_CUDA
#error Only one architecture must be defined
#endif
#endif

#ifndef BNN_ARCHITECTURE_CPU
#ifndef BNN_ARCHITECTURE_CUDA
#error One architecture must be defined
#endif
#endif

#ifdef BNN_ARCHITECTURE_CPU
#define BNN_LAMBDA_REFERENCE
#endif

#ifdef BNN_ARCHITECTURE_CUDA
#define BNN_LAMBDA_REFERENCE &
#endif

//#define debug_print(...) printf(__VA_ARGS__);
//#define debug_print_1(...) printf(__VA_ARGS__);
//#define debug_print(...)

typedef unsigned int u_word;
typedef signed int s_word;

#define QUANTITY_OF_BITS_IN_BYTE 8
#define QUANTITY_OF_BITS_IN_WORD (sizeof(u_word) * QUANTITY_OF_BITS_IN_BYTE)

#define LITTLE_TIME 1000

enum bnn_error_codes
{
    ok,
    error,
    invalid_value,
    out_of_range,
    malloc_fail
};

#endif // CONFIG_H
