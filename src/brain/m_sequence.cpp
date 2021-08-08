/*
 *   binary neurons network
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

m_sequence::m_sequence(unsigned int triggers_length)
{
    if((triggers_length < 2)||(triggers_length > sizeof (int) * 8 - 1))
        throw std::runtime_error("error");

    length = triggers_length;
}

bool m_sequence::next()
{
    unsigned int return_value = triggers & 1;

    triggers >>= 1;

    triggers |= ((triggers & 1) ^ return_value) << (length - 1);

    return return_value;
}

bool m_sequence::get_at(unsigned int future)
{
    return ((triggers >> future) & 1);
}

unsigned int m_sequence::get_registers()
{
    return triggers;
}

void m_sequence::set_triggers_length(unsigned int triggers_length)
{
    length = triggers_length;
}

} // namespace bnn
