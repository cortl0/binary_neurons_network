/*
 *   binary neurons network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef BNN_RANDOM_H
#define BNN_RANDOM_H

#include <random>

#include "../m_sequence.h"
#include "config.h"
#include "config.h"

namespace bnn::random
{

class random final
{
public:
    ~random();
    random() = delete;
    random(_word random_array_length_in_power_of_two, m_sequence& m_sequence);
    void put(bool i, config&) noexcept;
    _word get(_word bits, config&) noexcept;
    _word get_ft(_word from, _word to, config&) noexcept;
    _word get_length() const;
    _word get_under(_word to, config&) noexcept;
    std::vector<_word>& get_array();

private:
    std::vector<_word> array;

#ifdef DEBUG
public:
    unsigned long long int debug_count_get = 0;
    unsigned long long int debug_count_put = 0;
    long long int debug_sum_put = 0;
#endif
};

} // namespace bnn::random

#endif // BNN_RANDOM_H
