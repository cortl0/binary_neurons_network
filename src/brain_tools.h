/*
 *   Binary Neurons Network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
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
    brain_tools(_word random_array_length_in_power_of_two,
                 _word quantity_of_neurons_in_power_of_two,
                 _word input_length,
                 _word output_length,
                 _word threads_count_in_power_of_two = 0);
    void debug_out();
    const _word& get_iteration() const;
    bool load(std::ifstream&);
    void primary_filling();
    void resize(_word brainBits);
    bool save(std::ofstream&);
    void stop();

protected:
    const _word brain_save_load_length = 13 * sizeof(_word)
            + sizeof(random_config)
            + sizeof(m_sequence_);

    const _word thread_save_load_length = 6 * sizeof(_word)
        #ifdef DEBUG
            + 8 * sizeof(_word)
        #endif
            + sizeof(random_config);

private:
    std::thread thread_debug_out;
};

} // namespace bnn

#endif // BNN_BRAIN_TOOLS_H
