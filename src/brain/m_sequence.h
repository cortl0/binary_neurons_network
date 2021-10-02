/*
 *   binary neurons network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef M_SEQUENCE_H
#define M_SEQUENCE_H

#include <stdexcept>
#include "config.h"

namespace bnn
{

class m_sequence
{
    unsigned int triggers = 1;
    unsigned int length;
public:
    m_sequence();
    m_sequence(unsigned int triggers_length);
    bool next();
    bool get_at(unsigned int future);
    unsigned int get_registers();
    void set_triggers_length(unsigned int triggers_length);
};

} // namespace bnn

#endif // M_SEQUENCE_H
