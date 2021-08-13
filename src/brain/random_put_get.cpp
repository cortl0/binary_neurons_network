/*
 *   binary neurons network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#include "random_put_get.h"

namespace bnn
{

random_put_get::~random_put_get() { }

random_put_get::random_put_get(_word random_array_length_in_power_of_two, m_sequence& m_sequence)
{
    length = (1 << random_array_length_in_power_of_two) / _word_bits;
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
    for (_word i = 0; i < length; i++)
        for (_word j = 0; j < _word_bits; j++)
            put(m_sequence.next());
#endif
}

void random_put_get::put(bool i) noexcept
{
#ifdef DEBUG
    debug_sum_put += i * 2 - 1;
    debug_count_put++;
#endif
    array[offset] = (array[offset] & (~(1 << offset_in_word))) | (static_cast<_word>(i) << offset_in_word);
    offset_in_word++;
    if (offset_in_word >= _word_bits)
    {
        offset_in_word = 0;
        offset++;
        if (offset >= length)
            offset = 0;
    }
}

_word random_put_get::get(_word bits) noexcept
{
#ifdef DEBUG
    debug_count_get += bits;
#endif
    _word returnValue = array[offset] >> offset_in_word;
    if(bits > _word_bits - offset_in_word)
    {
        returnValue = returnValue & (~((~static_cast<_word>(0)) << (_word_bits - offset_in_word)));
        offset_in_word = offset_in_word + bits - _word_bits;
        offset++;
        if (offset >= length)
            offset = 0;
        returnValue = returnValue | ((array[offset] & (~((~static_cast<_word>(0)) << offset_in_word))) << (bits - offset_in_word));
    }
    else
    {
        returnValue = returnValue & (~((~static_cast<_word>(0)) << bits));
        offset_in_word += bits;
        if (offset_in_word >= _word_bits)
        {
            offset_in_word -= _word_bits;
            offset++;
            if (offset >= length)
                offset = 0;
        }
    }
    return returnValue;
}

_word random_put_get::get_ft(_word from, _word to) noexcept
{
    _word returnValue, i = 1, j = 2, tmf;
    tmf = to - from;
    while (true) { if (j > tmf) break; i++; j = j << 1; }
    while (true)
    {
        returnValue = get(i);
        if (returnValue <= tmf)
            return returnValue + from;
    }
}

_word random_put_get::get_length()
{
    return length;
}

_word random_put_get::get_under(_word to) noexcept
{
    _word count = 0;

    while(to >>= 1)
    {
        count++;
    }

    return get(count);
}

std::vector<_word>& random_put_get::get_array()
{
    return array;
}

} // namespace bnn
