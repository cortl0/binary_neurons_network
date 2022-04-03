/*
 *   binary neurons network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#include "random.h"

namespace bnn::random
{

random::~random() { }

random::random(u_word random_array_length_in_power_of_two)
{
    u_word length = (1 << random_array_length_in_power_of_two) / QUANTITY_OF_BITS_IN_WORD;
    array.resize(length);

    config config_;
    config_.put_offset_start = 0;
    config_.put_offset = config_.put_offset_start;
    config_.put_offset_end = array.size();

#define fill_from 2
#if(fill_from == 0)
    // fill the array with random numbers
    for (_word i = 0; i < length; i++)
        for (_word j = 0; j < _word_bits; j++)
            put(rand() % 2);
#elif(fill_from == 1)
    // fill the array with random numbers Mersenne Twister
    std::mt19937 gen;
    std::uniform_int_distribution<> uid = std::uniform_int_distribution<>(0, 1);
    for (_word i = 0; i < length; i++)
        for (_word j = 0; j < QUANTITY_OF_BITS_IN_WORD; j++)
            put(uid(gen), config_);
#elif(fill_from == 2)
    // fill the array with M-sequence
    // no need to use random number algorithms
    m_sequence m_sequence_(random_array_length_in_power_of_two >= QUANTITY_OF_BITS_IN_WORD ?
                           QUANTITY_OF_BITS_IN_WORD - 1 : random_array_length_in_power_of_two);

    for (u_word i = 0; i < array.size(); i++)
        for (u_word j = 0; j < QUANTITY_OF_BITS_IN_WORD; j++)
            put(m_sequence_.next(), config_);
#endif
}

void random::put(const bool i, config& config_) noexcept(true)
{
#ifdef DEBUG
    config_.debug_sum_put += i * 2 - 1;
    config_.debug_count_put++;
#endif
    array[config_.put_offset] = (array[config_.put_offset] & (~(1 << config_.put_offset_in_word))) | (static_cast<u_word>(i) << config_.put_offset_in_word);
    config_.put_offset_in_word++;
    if (config_.put_offset_in_word >= QUANTITY_OF_BITS_IN_WORD)
    {
        config_.put_offset_in_word = 0;
        config_.put_offset++;
        if (config_.put_offset >= config_.put_offset_end)
            config_.put_offset = 0;
    }
}

u_word random::get(u_word bits, config& config_) noexcept(true)
{
#ifdef DEBUG
    config_.debug_count_get += bits;
#endif

    u_word shift, quantity, data, returnValue = 0;

    while(bits)
    {
        quantity = QUANTITY_OF_BITS_IN_WORD - config_.get_offset_in_word;
        quantity = quantity < bits ? quantity : bits;
        data = array[config_.get_offset] >> config_.get_offset_in_word;
        shift = QUANTITY_OF_BITS_IN_WORD - quantity;
        data <<= shift;
        data >>= shift;
        returnValue <<= quantity;
        returnValue |= data;
        bits -= quantity;
        config_.get_offset_in_word += quantity;
        if (config_.get_offset_in_word >= QUANTITY_OF_BITS_IN_WORD)
        {
            config_.get_offset_in_word -= QUANTITY_OF_BITS_IN_WORD;
            config_.get_offset++;
            if (config_.get_offset >= array.size())
                config_.get_offset = 0;
        }
    }
    return returnValue;
}

u_word random::get_ft(const u_word from, const u_word to, config& config_) noexcept(true)
{
    u_word returnValue, i = 1, j = 2, tmf;
    tmf = to - from;
    while (true) { if (j > tmf) break; i++; j = j << 1; }
    while (true)
    {
        returnValue = get(i, config_);
        if (returnValue <= tmf)
            return returnValue + from;
    }
}

u_word random::get_length() const
{
    return array.size();
}

u_word random::get_under(u_word to, config& config_) noexcept(true)
{
    u_word count = 0;

    while(to >>= 1)
        count++;

    return get(count, config_);
}

std::vector<u_word>& random::get_array()
{
    return array;
}

} // namespace bnn::random
