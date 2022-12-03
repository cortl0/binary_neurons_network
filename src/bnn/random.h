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

#include "../common/headers/config.hpp"

namespace bnn::random
{

class random final
{
public:
    struct config
    {
        u_word get_offset = 0;
        u_word get_offset_in_word = 0;
        u_word put_offset_start;
        u_word put_offset_end;
        u_word put_offset;
        u_word put_offset_in_word = 0;

    #ifdef DEBUG
        unsigned long long int debug_count_get = 0;
        unsigned long long int debug_count_put = 0;
        long long int debug_sum_put = 0;
    #endif
    };

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
