/*
 *   Binary Neurons Network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef BNN_BRAIN_TOOLS_H
#define BNN_BRAIN_TOOLS_H

#include <experimental/filesystem>
#include <fstream>
#include <stdexcept>

#include "bnn.h"

namespace fs = std::experimental::filesystem;

namespace bnn
{

class brain_tools : public brain
{
public:
    virtual ~brain_tools();
    brain_tools() = delete;
    brain_tools(u_word quantity_of_neurons_in_power_of_two,
                u_word input_length,
                u_word output_length,
                u_word threads_count_in_power_of_two = 0);
    void get_debug_string(std::string&);
    const u_word& get_iteration() const;
    bool load(std::ifstream&);
    void primary_filling();
    void resize(u_word brainBits);
    bool save(std::ofstream&);
    void save_random();
    void save_random_bin();
    void save_random_csv();
    void save_random_csv_line();
    void stop();
};

} // namespace bnn

#endif // BNN_BRAIN_TOOLS_H
