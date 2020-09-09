//*************************************************************//
//                                                             //
//   binary neurons network                                    //
//   created by Ilya Shishkin                                  //
//   cortl@8iter.ru                                            //
//   http://8iter.ru/ai.html                                   //
//   https://github.com/cortl0/binary_neurons_network          //
//   licensed by GPL v3.0                                      //
//                                                             //
//*************************************************************//

#ifndef RANDOM_PUT_GET_H
#define RANDOM_PUT_GET_H

#include <random>

#include "pch.h"

namespace bnn
{

class random_put_get
{
    _word offset = 0;
    _word offset_in_word = 0;
    _word length;
    std::vector<_word> array;
    std::mt19937 gen;
    std::uniform_int_distribution<> uid = std::uniform_int_distribution<>(0, 1);
public:
    _word debug_count_put=0;
    _word debug_count_get=0;
    ~random_put_get();
    random_put_get() = delete;
    random_put_get(_word random_array_length_in_power_of_two);
    void put(bool i) noexcept;
    _word get(_word bits) noexcept;
    _word get_ft(_word from, _word to) noexcept;
    _word get_length();
    std::vector<_word>& get_array();
};

} // !namespace bnn

#endif // RANDOM_PUT_GET_H
