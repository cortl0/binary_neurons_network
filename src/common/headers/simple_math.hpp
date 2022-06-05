/*
 *   Binary Neurons Network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef BNN_SIMPLE_MATH_HPP
#define BNN_SIMPLE_MATH_HPP

#include "config.hpp"

namespace bnn
{

struct simple_math final
{
    inline static s_word sign(int i) noexcept(true) { if (i < 0) return (-1); else return (1); }
    inline static s_word sign0(int i) noexcept(true) { if (i < 0) return (-1); else if (i > 0) return (1); else return (0); }
    inline static u_word two_pow_x(u_word x) noexcept(true) { return 1 << static_cast<s_word>(x); }
    inline static s_word abs(int i) noexcept(true) { if (i < 0) return (-i); else return (i); }
    
    inline static u_word log2_1(u_word x)
    {
        if(!x)
            throw_error(" x = 0");

        u_word result = -1;

        while(x)
        {
            x >>= 1;
            result++;
        }

        return result;
    }
};

} // namespace bnn

#endif // BNN_SIMPLE_MATH_HPP
