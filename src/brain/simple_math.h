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

#include "config.h"

namespace bnn
{

struct simple_math final
{
    inline static int sign(int i) noexcept { if (i < 0) return (-1); else return (1); }
    inline static int sign0(int i) noexcept { if (i < 0) return (-1); else if (i > 0) return (1); else return (0); }
    inline static _word two_pow_x(_word x) noexcept { return 1 << static_cast<int>(x); }
    inline static int abs(int i) noexcept { if (i < 0) return (-i); else return (i); }
    
    inline static _word log2_1(_word x)
    {
		if(!x)
			throw __LINE__ + " x = 0";
			
		_word result = -1;
		
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
