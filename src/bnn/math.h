/*
 *   Binary Neurons Network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef BNN_MATH_H
#define BNN_MATH_H

#include "config.h"

auto bnn_math_sign = [BNN_LAMBDA_REFERENCE](s_word i) -> s_word
{
    if(i < 0)
        return s_word{-1};
    else
        return s_word{1};
};

auto bnn_math_sign0 = [BNN_LAMBDA_REFERENCE](s_word i) -> s_word
{
    if(i < 0)
        return s_word{-1};
    else if(i > 0)
        return s_word{1};
    else
        return s_word{0};
};

auto bnn_math_two_pow_x = [BNN_LAMBDA_REFERENCE](u_word x) -> u_word
{
    return u_word(1) << x;
};

auto bnn_math_abs = [BNN_LAMBDA_REFERENCE](s_word i) -> s_word
{
    if(i < 0)
        return -i;

    return i;
};

auto bnn_math_log2_1 = [BNN_LAMBDA_REFERENCE](
        bnn_error_codes* bnn_error_code,
        u_word x
        ) -> u_word
{
    if(!x)
        *bnn_error_code = bnn_error_codes::invalid_value;

    u_word result = ~u_word{0};

    while(x)
    {
        x >>= 1;
        ++result;
    }

    return result;
};

#endif // BNN_MATH_H
