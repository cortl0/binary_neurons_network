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

#include "brain/brain.h"

namespace fs = std::experimental::filesystem;

namespace bnn
{

struct brain_tools : public brain
{
public:
    virtual ~brain_tools();
    brain_tools() = delete;
    brain_tools(u_word quantity_of_neurons_in_power_of_two,
                u_word input_length,
                u_word output_length,
                u_word threads_count_in_power_of_two = 0);
    void debug_out();
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

protected:
    const u_word brain_save_load_length = 13 * sizeof(u_word)
            + sizeof(random_config);

    const u_word thread_save_load_length = 6 * sizeof(u_word)
        #ifdef DEBUG
            + 8 * sizeof(u_word)
        #endif
            + sizeof(random_config);

private:
    std::thread thread_debug_out;
};

} // namespace bnn

#endif // BNN_BRAIN_TOOLS_H
