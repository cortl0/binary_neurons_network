/*
 *   Binary Neurons Network
 *   created by Ilya Shishkin
 *   cortl@yandex.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef BNN_GPU_CUDA_CUDA_H
#define BNN_GPU_CUDA_CUDA_H

#include "bnn/state.h"
#include "common/settings.h"
#include "memory.hpp"

#ifndef BNN_ARCHITECTURE_CUDA
#define BNN_ARCHITECTURE_CUDA
#endif

struct bnn_bnn;

namespace bnn::gpu
{

class cuda
{
public:
    ~cuda();
    cuda(const bnn_settings&);
    void calculate_pointers();
    bool get_output(u_word offset);
    bnn_state get_state();
    void set_input(u_word offset, bool value);
    static bool allocate_host_and_device_memory(memory&);
    static bool free_host_and_device_memory(memory&);
    void initialize();
    bool memory_copy_host_to_device(memory&);
    bool memory_copy_device_to_host(memory&);
    void start();
    void stop();

protected:
    void upload();
    void download();

    bnn_bnn* bnn{nullptr};

private:
    void run();
    bool test_kernel_result();
    bnn_bnn* bnn_host{nullptr};
    memory memory_;
    memory debug_memory;
    bnn_state state{bnn_state::stopped};
    bool one_time_trigger{true};
    bool one_time_trigger_stop{};
};

} // namespace bnn::gpu

#endif // BNN_GPU_CUDA_CUDA_H
