/*
 *   binary neurons network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef BNN_BRAIN_FRIEND_H
#define BNN_BRAIN_FRIEND_H

#include <list>
#include <experimental/filesystem>
#include <fstream>
#include <stdexcept>

#include "brain/brain.h"

namespace fs = std::experimental::filesystem;

namespace bnn
{

struct brain_friend
{
    bnn::brain &brain_;

    brain_friend() = delete;
    brain_friend(bnn::brain &);
    static void debug_out(brain*, _word &old_iteration);
    bool load(std::ifstream&);
    void resize(_word brainBits);
    bool save(std::ofstream&);
    void stop();
};

} // namespace bnn

#endif // BNN_BRAIN_FRIEND_H
