/*
 *   Binary Neurons Network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef BNN_GPU_CUDA_MEMORY_H
#define BNN_GPU_CUDA_MEMORY_H

namespace bnn::gpu::cuda
{

struct memory
{
    int* host_data{nullptr};
    int* device_data{nullptr};
    int size{0};
} memory_;

} // namespace bnn::gpu::cuda

#endif // BNN_GPU_CUDA_MEMORY_H
