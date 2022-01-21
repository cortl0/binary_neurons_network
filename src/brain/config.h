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

#define DEBUG

#define QUANTITY_OF_BITS_IN_BYTE 8

#define _word unsigned int

#define QUANTITY_OF_BITS_IN_WORD (sizeof(_word) * QUANTITY_OF_BITS_IN_BYTE)

#endif // BNN_CONFIG_H
