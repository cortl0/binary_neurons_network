/*
 *   Binary Neurons Network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef BNN_COMMON_BNN_TOOLS_H
#define BNN_COMMON_BNN_TOOLS_H

#include <fstream>
#include <stdexcept>

#include "architecture.h"

#include "bnn/bnn.h"

namespace bnn
{

class brain_tools : public architecture
{
public:
    virtual ~brain_tools();
    brain_tools(const bnn_settings&);
    void get_debug_string(std::string&);
    const u_word& get_iteration() const;
    bool load(std::ifstream&);
#if(0)
    void primary_filling();
#endif
    void resize(u_word brainBits);
    bool save(std::ofstream&);
    void save_random();
    void save_random_bin();
    void save_random_csv();
    void save_random_csv_line();
    void stop();
};

} // namespace bnn

#endif // BNN_COMMON_BNN_TOOLS_H
