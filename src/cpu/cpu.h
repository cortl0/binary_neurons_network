/*
 *   Binary Neurons Network
 *   created by Ilya Shishkin
 *   cortl@yandex.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef BNN_CPU_CPU_H
#define BNN_CPU_CPU_H

#include <thread>
#include <vector>

#include "bnn/state.h"
#include "common/settings.h"

#ifndef BNN_ARCHITECTURE_CPU
#define BNN_ARCHITECTURE_CPU
#endif

struct bnn_bnn;

namespace bnn
{

class cpu
{
public:
    ~cpu();
    cpu(const bnn_settings&);
    void calculate_pointers();
    void start();
    void stop();
    void set_input(u_word i, bool value);
    bool get_output(u_word i);
    bnn_state get_state();
    void initialize() {}

protected:
    void upload() {}
    void download() {}

    bnn_bnn* bnn{nullptr};

private:
    std::thread main_thread;
    std::vector<std::thread> treads;
};

} // namespace bnn

#endif // BNN_CPU_CPU_H
