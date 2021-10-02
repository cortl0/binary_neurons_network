/*
 *   binary neurons network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef STATE_H
#define STATE_H

namespace bnn
{

enum state
{
    stopped = 0,
    start = 1,
    started = 2,
    stop = 3
};

} // namespace bnn

#endif // STATE_H
