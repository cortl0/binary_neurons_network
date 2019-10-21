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

#include "random_put_get.h"

random_put_get::~random_put_get()
{
    delete [] array;
}
random_put_get::random_put_get(_word array_length_in_bits)
{
    length = (1 << (array_length_in_bits - 1)) / _word_bits;
    array = new _word[length];
    for (_word i = 0; i < length; i++)
        for (_word j = 0; j < _word_bits; j++)
        {
            //RndmPut(rand() % 2);
            put(uid(gen));
        }
}
void random_put_get::put(bool i) noexcept
{
    debug_count_put++;
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
    debug_count_get += bits;
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
_word* random_put_get::get_array()
{
    return array;
}
