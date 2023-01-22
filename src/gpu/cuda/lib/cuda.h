/*
 *   Binary Neurons Network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#include <thread>

#include "bnn/bnn_settings.h"
#include "memory.hpp"

#define BNN_ARCHITECTURE_CUDA

#include "bnn/bnn.h"

namespace bnn::gpu
{

class cuda
{
public:
    ~cuda();
    cuda(const bnn_settings& bs);

    void allocate_host_and_device_memory(memory& m);

    void free_host_and_device_memory();

    void memory_copy_host_to_device(memory& m);

    void memory_copy_host_to_device();

    void memory_copy_device_to_host(memory& m);

    void memory_copy_device_to_host();

    void start();

    void stop();

    bool is_active();

    static void run(cuda* me);

private:
    bool active{false};
    bnn_bnn* bnn_host{nullptr};
    memory memory_;
    memory debug_memory;
    std::thread thread;

};

} // namespace bnn::gpu
