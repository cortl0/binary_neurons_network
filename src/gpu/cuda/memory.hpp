#ifndef BNN_GPU_CUDA_MEMORY_H
#define BNN_GPU_CUDA_MEMORY_H

namespace bnn::gpu::cuda
{

struct memory
{
    int *host_data{nullptr};
    int *device_data{nullptr};
    int size{0};
} memory_;

} // namespace bnn::gpu::cuda

#endif // BNN_GPU_CUDA_MEMORY_H
