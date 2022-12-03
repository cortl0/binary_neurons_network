#ifndef BNN_GPU_CUDA_GPU_HPP
#define BNN_GPU_CUDA_GPU_HPP

#include "helper_cuda.h"

#include "memory.hpp"
#include "statistic.hpp"
#include "settings.hpp"

namespace bnn::gpu::cuda
{

__device__ void fff(int *g_data, int inc_value)
{

#include "c.h"
    B b;
    b.a.x;
    foooooo();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = inc_value; i; i--)
        g_data[idx]++;
}

__global__ void increment_kernel(int *g_data, int inc_value)
{
    fff(g_data, inc_value);
}

class gpu
{
public:
    static void allocate_host_and_device_memory(memory& m)
    {
        checkCudaErrors(cudaMallocHost((void **)&m.host_data, m.size));
        memset(m.host_data, 0, m.size);

        checkCudaErrors(cudaMalloc((void **)&m.device_data, m.size));
        checkCudaErrors(cudaMemset(m.device_data, 0, m.size));
    }

    static void free_host_and_device_memory(memory& m)
    {
        checkCudaErrors(cudaFreeHost(m.host_data));
        checkCudaErrors(cudaFree(m.device_data));
    }

    static void memory_copy_host_to_device(memory& m)
    {
        cudaMemcpyAsync(m.device_data, m.host_data, m.size, cudaMemcpyHostToDevice, 0);
    }

    static void memory_copy_device_to_host(memory& m)
    {
        cudaMemcpyAsync(m.host_data, m.device_data, m.size, cudaMemcpyDeviceToHost, 0);
    }

    static bool correct_output(int *data, const int n, const int x)
    {
        uint64_t u = 0;

        for (int i = 0; i < n; i++)
        {
            if (data[i] != x) {
                printf("Error! data[%d] = %d, ref = %d\n", i, data[i], x);
                return false;
            }
            u += data[i];
        }

        printf("[%llu] cycles\n", u);

        return true;
    }

    static void run(memory memory_, int value, int n)
    {
        // This will pick the best possible CUDA capable device
        const char** argv = nullptr;
        int devID = findCudaDevice(0, (const char **)argv);
        cudaDeviceProp cdp;
        checkCudaErrors(cudaGetDeviceProperties(&cdp, devID));

        gpu::allocate_host_and_device_memory(memory_);

        // set kernel launch configuration
        dim3 threads = dim3(settings::calculate_thread_dim_x(cdp, n));
        dim3 blocks = dim3(settings::calculate_block_dim_x(cdp, n, threads));

        printf("threads count [%d]\n", threads.x * blocks.x);

        // create cuda event handles
        cudaEvent_t start, stop;
        checkCudaErrors(cudaEventCreate(&start));
        checkCudaErrors(cudaEventCreate(&stop));

        checkCudaErrors(cudaDeviceSynchronize());

        // asynchronously issue work to the GPU (all to stream 0)
        cudaEventRecord(start, 0);

        memory_copy_host_to_device(memory_);
        increment_kernel<<<blocks, threads, 0, 0>>>(memory_.device_data, value);
        memory_copy_device_to_host(memory_);

        cudaEventRecord(stop, 0);

        // have CPU do some work while waiting for stage 1 to finish
        unsigned long int counter = 0;

        while (cudaEventQuery(stop) == cudaErrorNotReady) {
            usleep(1000000);
            //cudaMemcpyAsync(a, d_a, nbytes, cudaMemcpyDeviceToHost, 0);
            counter++;
        }

        printf("CPU executed %lu iterations while waiting for GPU to finish\n", counter);

        // check the output for correctness
        bool bFinalResults = gpu::correct_output(memory_.host_data, n, value);

        // release resources
        checkCudaErrors(cudaEventDestroy(start));
        checkCudaErrors(cudaEventDestroy(stop));
        gpu::free_host_and_device_memory(memory_);

        exit(bFinalResults ? EXIT_SUCCESS : EXIT_FAILURE);
    }
};

} // namespace bnn::gpu::cuda

#endif // BNN_GPU_CUDA_GPU_HPP
