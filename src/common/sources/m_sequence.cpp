/*
 *   Binary Neurons Network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#include "../headers/m_sequence.h"

#include <stdexcept>

namespace bnn
{

m_sequence::m_sequence(u_word length)
    : length(length)
{
    if((length < 2) || (length > QUANTITY_OF_BITS_IN_WORD - 1))
        throw std::range_error(
                log_string("invalid value ["
                           + std::to_string(length)
                           + "] of triggers_length"));
}

bool m_sequence::next()
{
    u_word return_value = triggers & 1;

    triggers >>= 1;

    triggers |= ((triggers & 1) ^ return_value) << (length - 1);

    return return_value;
}

} // namespace bnn
