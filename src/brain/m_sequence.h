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

#include "config.hpp"

namespace bnn
{

class m_sequence final
{
public:
    m_sequence() = delete;
    explicit m_sequence(u_word length);
    bool next();

private:
    u_word triggers = 1;
    u_word length;
};

} // namespace bnn

#endif // BNN_M_SEQUENCE_H
