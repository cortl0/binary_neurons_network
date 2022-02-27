/*
 *   Binary Neurons Network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef BNN_M_SEQUENCE_H
#define BNN_M_SEQUENCE_H

#include <stdexcept>
#include "config.hpp"

namespace bnn
{

class m_sequence final
{
    u_word triggers = 1;
    u_word length;
public:
    m_sequence();
    m_sequence(u_word triggers_length);
    bool next();
    bool get_at(u_word future);
    u_word get_registers();
    void set_triggers_length(u_word triggers_length);
};

} // namespace bnn

#endif // BNN_M_SEQUENCE_H
