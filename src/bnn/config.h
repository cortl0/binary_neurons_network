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

#ifndef DEBUG
#define DEBUG
#endif

//#include <iostream>

//#define bnn_debug_print(...) printf(__VA_ARGS__);
//#define bnn_debug_print_1(...) printf(__VA_ARGS__);
//#define bnn_debug_print(...)

typedef unsigned int u_word;
typedef signed int s_word;

#define QUANTITY_OF_BITS_IN_BYTE 8
#define QUANTITY_OF_BITS_IN_WORD (sizeof(u_word) * QUANTITY_OF_BITS_IN_BYTE)

#define BNN_LITTLE_TIME 1000

#endif // BNN_CONFIG_H
