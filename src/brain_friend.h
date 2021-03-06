/*
 *   binary neurons network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef BRAIN_FRIEND_H
#define BRAIN_FRIEND_H

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
    brain_friend(bnn::brain &brain_);
    bool load(std::ifstream&);
    void resize(_word brainBits);
    bool save(std::ofstream&);
    void stop();
};

} // namespace bnn

#endif // BRAIN_FRIEND_H
