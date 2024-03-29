/*
 *   Binary Neurons Network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef BNN_COMMON_STATE_H
#define BNN_COMMON_STATE_H

namespace bnn
{

enum class state : int
{
    start = 1,
    started,
    stop,
    stopped
};

} // namespace bnn

#endif // BNN_COMMON_STATE_H
