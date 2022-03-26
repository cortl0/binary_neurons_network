/*
 *   Binary Neurons Network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#include "m_sequence.h"

namespace bnn
{

m_sequence::m_sequence()
{
    length = 3;
}

m_sequence::m_sequence(u_word triggers_length)
{
    if((triggers_length < 2) || (triggers_length > QUANTITY_OF_BITS_IN_WORD - 1))
        throw_error("error");

    length = triggers_length;
}

bool m_sequence::next()
{
    u_word return_value = triggers & 1;

    triggers >>= 1;

    triggers |= ((triggers & 1) ^ return_value) << (length - 1);

    return return_value;
}

bool m_sequence::get_at(u_word future)
{
    return (triggers >> future) & 1;
}

u_word m_sequence::get_registers()
{
    return triggers;
}

void m_sequence::set_triggers_length(u_word triggers_length)
{
    length = triggers_length;
}

} // namespace bnn
