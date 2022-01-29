/*
 *   Binary Neurons Network
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

struct brain_friend : brain
{
    virtual ~brain_friend();
    brain_friend() = delete;
    brain_friend(_word random_array_length_in_power_of_two,
                 _word quantity_of_neurons_in_power_of_two,
                 _word input_length,
                 _word output_length,
                 _word threads_count_in_power_of_two = 0);
    void debug_out();
    bool load(std::ifstream&);
    void resize(_word brainBits);
    bool save(std::ofstream&);
    void stop();

private:
    std::thread thread_debug_out;
};

} // namespace bnn

#endif // BNN_BRAIN_FRIEND_H
