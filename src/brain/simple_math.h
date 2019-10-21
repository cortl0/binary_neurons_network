//*************************************************************//
//                                                             //
//   network of binary neurons                                 //
//   created by Ilya Shishkin                                  //
//   cortl@8iter.ru                                            //
//   http://8iter.ru/ai.html                                   //
//   https://github.com/cortl0/network_of_binary_neurons_cpp   //
//   licensed by GPL v3.0                                      //
//                                                             //
//*************************************************************//

#ifndef SIMPLE_MATH_H
#define SIMPLE_MATH_H

#include "pch.h"

struct simple_math
{
    inline static int sign(int i) noexcept { if (i < 0) return (-1); else return (1); }
    inline static int sign0(int i) noexcept { if (i < 0) return (-1); else if (i > 0) return (1); else return (0); }
    inline static _word two_pow_x(_word x) noexcept { return 1 << static_cast<int>(x); }
    inline static int abs(int i) noexcept { if (i < 0) return (-i); else return (i); }
};

#endif // SIMPLE_MATH_H
