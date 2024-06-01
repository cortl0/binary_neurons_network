/*
 *   Binary Neurons Network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef BNN_M_SEQUENCE_H
#define BNN_M_SEQUENCE_H

#include "config.h"

struct bnn_m_sequence
{
    u_word triggers = 1;
    u_word length;
    bnn_error_codes bnn_error_code{bnn_error_codes::ok};
};

auto bnn_m_sequence_set = [BNN_LAMBDA_REFERENCE](
        bnn_m_sequence* me,
        u_word length
        ) -> void
{
    if(length < 2)
    {
        me->bnn_error_code = bnn_error_codes::invalid_value;
        length = 2;
    }

    if(length > QUANTITY_OF_BITS_IN_WORD)
    {
        me->bnn_error_code = bnn_error_codes::invalid_value;
        length = QUANTITY_OF_BITS_IN_WORD;
    }

    me->length = length;
};

auto bnn_m_sequence_next = [BNN_LAMBDA_REFERENCE](
        bnn_m_sequence* me
        ) -> bool
{
    u_word return_value = me->triggers & 1;
    me->triggers >>= 1;
    me->triggers |= ((me->triggers & 1) ^ return_value) << (me->length - 1);
    return return_value;
};

#endif // BNN_M_SEQUENCE_H
