/*
 *   Binary Neurons Network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef BNN_STATE_HPP
#define BNN_STATE_HPP

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

#endif // BNN_STATE_HPP