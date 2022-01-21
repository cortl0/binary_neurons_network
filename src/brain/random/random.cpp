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

random::random(_word random_array_length_in_power_of_two, m_sequence& m_sequence)
{
    _word length = (1 << random_array_length_in_power_of_two) / QUANTITY_OF_BITS_IN_WORD;
    array.resize(length);

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
        for (_word j = 0; j < _word_bits; j++)
            put(uid(gen));
#elif(fill_from == 2)
    // fill the array with M-sequence
    // no need to use random number algorithms
    config config_;
    config_.put_offset_start = 0,
    config_.put_offset_end = length;
    for (_word i = 0; i < length; i++)
        for (_word j = 0; j < QUANTITY_OF_BITS_IN_WORD; j++)
            put(m_sequence.next(), config_);
#endif
}

void random::put(bool i, config& config_) noexcept
{
#ifdef DEBUG
    debug_sum_put += i * 2 - 1;
    debug_count_put++;
#endif
    _word offset = config_.put_offset_start + config_.put_offset;
    array[offset] = (array[offset] & (~(1 << config_.put_offset_in_word))) | (static_cast<_word>(i) << config_.put_offset_in_word);
    config_.put_offset_in_word++;
    if (config_.put_offset_in_word >= QUANTITY_OF_BITS_IN_WORD)
    {
        config_.put_offset_in_word = 0;
        config_.put_offset++;
        if (offset >= config_.put_offset_end)
            config_.put_offset = 0;
    }
}

_word random::get(_word bits, config& config_) noexcept
{
#ifdef DEBUG
    debug_count_get += bits;
#endif
    _word returnValue = array[config_.get_offset] >> config_.get_offset_in_word;
    if(bits > QUANTITY_OF_BITS_IN_WORD - config_.get_offset_in_word)
    {
        returnValue = returnValue & (~((~static_cast<_word>(0)) << (QUANTITY_OF_BITS_IN_WORD - config_.get_offset_in_word)));
        config_.get_offset_in_word = config_.get_offset_in_word + bits - QUANTITY_OF_BITS_IN_WORD;
        config_.get_offset++;
        if (config_.get_offset >= array.size())
            config_.get_offset = 0;
        returnValue = returnValue | ((array[config_.get_offset] & (~((~static_cast<_word>(0)) << config_.get_offset_in_word))) << (bits - config_.get_offset_in_word));
    }
    else
    {
        returnValue = returnValue & (~((~static_cast<_word>(0)) << bits));
        config_.get_offset_in_word += bits;
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

//_word random_put_get::get_ft(_word from, _word to) noexcept
//{
//    _word returnValue, i = 1, j = 2, tmf;
//    tmf = to - from;
//    while (true) { if (j > tmf) break; i++; j = j << 1; }
//    while (true)
//    {
//        returnValue = get(i);
//        if (returnValue <= tmf)
//            return returnValue + from;
//    }
//}

_word random::get_length()
{
    return array.size();
}

_word random::get_under(_word to, config& config_) noexcept
{
    _word count = 0;

    while(to >>= 1)
    {
        count++;
    }

    return get(count, config_);
}

std::vector<_word>& random::get_array()
{
    return array;
}

} // namespace bnn::random
