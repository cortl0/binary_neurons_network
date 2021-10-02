/*
 *   binary neurons network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef BNN_CONFIG_H
#define BNN_CONFIG_H

#define DEBUG

#define bits_in_byte 8

//#define _word uint16_t;
//#define _word uint32_t;
//#define _word uint64_t;
#define _word unsigned int

#define _word_bits (sizeof(_word) * bits_in_byte)

#endif // BNN_CONFIG_H
