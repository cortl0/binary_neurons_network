/*
 *   binary neurons network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef BNN_RANDOM_RANDOM_H
#define BNN_RANDOM_RANDOM_H

#include <vector>

#include "config.hpp"
#include "../m_sequence.h"

namespace bnn::random
{

class random final
{
public:
    ~random();
    random() = delete;
    explicit random(u_word random_array_length_in_power_of_two);
    void put(bool i, config&) noexcept(true);
    u_word get(u_word bits, config&) noexcept(true);
    u_word get_ft(u_word from, u_word to, config&) noexcept(true);
    u_word get_length() const;
    u_word get_under(u_word to, config&) noexcept(true);
    std::vector<u_word>& get_array();

private:
    std::vector<u_word> array;
};

} // namespace bnn::random

#endif // BNN_RANDOM_RANDOM_H
