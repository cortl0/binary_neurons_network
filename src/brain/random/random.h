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
    void put(bool i, config&) noexcept(true);
    _word get(_word bits, config&) noexcept(true);
    _word get_ft(_word from, _word to, config&) noexcept(true);
    _word get_length() const;
    _word get_under(_word to, config&) noexcept(true);
    std::vector<_word>& get_array();

private:
    std::vector<_word> array;
};

} // namespace bnn::random

#endif // BNN_RANDOM_H
