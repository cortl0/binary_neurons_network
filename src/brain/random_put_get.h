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

#include "m_sequence.h"
#include "config.h"

namespace bnn
{

class random_put_get
{
    _word offset = 0;
    _word offset_in_word = 0;
    _word length;
    std::vector<_word> array;
public:
    unsigned long long int debug_count_put=0;
    unsigned long long int debug_count_get=0;
    ~random_put_get();
    random_put_get() = delete;
    random_put_get(_word random_array_length_in_power_of_two, _word random_max_value_to_fill_in_power_of_two);
    void put(bool i) noexcept;
    _word get(_word bits) noexcept;
    _word get_ft(_word from, _word to) noexcept;
    _word get_length();
    std::vector<_word>& get_array();
};

} // namespace bnn

#endif // RANDOM_PUT_GET_H
