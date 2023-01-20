/*
 *   Binary Neurons Network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef BNN_GPU_CUDA_GPU_HPP
#define BNN_GPU_CUDA_GPU_HPP

#include <thread>

#include "helper_cuda.h"

#include "memory.hpp"
#include "statistic.hpp"
#include "settings.hpp"

#include "bnn/bnn_settings.h"

#define BNN_ARCHITECTURE_CUDA

#include "../../bnn/bnn.h"

__global__ void increment_kernel(int* g_data, int* debug_data)
{
#include "undef_implementations.h"
#include "../../bnn/bnn_implementation.h"

    int thread_number = blockIdx.x * blockDim.x + threadIdx.x;

    bnn_bnn* bnn = (bnn_bnn*)g_data;
    bnn_thread_function(bnn, thread_number);
    for(u_word thread_number = 0; thread_number < bnn->threads_.size; ++thread_number)
        debug_data[thread_number] = bnn->threads_.data[thread_number].iteration;
}

namespace bnn::gpu::cuda
{

class gpu
{
public:
    ~gpu()
    {
        stop();
        while(is_active());
        free_host_and_device_memory();
        printf("========================================== ~gpu()\n");
    }
    gpu(const bnn_settings& bs)
    {
#include "undef_implementations.h"
#include "../../bnn/bnn_implementation.h"

        debug_memory.size = 1024;
        printf("========================================== allocate_host_and_device_memory(debug_memory)\n");
        allocate_host_and_device_memory(debug_memory);
        bnn_bnn bnn_temp;
        bnn_temp.storage_.size_in_power_of_two = bs.quantity_of_neurons_in_power_of_two;
        bnn_temp.input_.size = bs.input_length;
        bnn_temp.output_.size = bs.output_length;
        bnn_temp.threads_.size_in_power_of_two = bs.threads_count_in_power_of_two;
        bnn_calculate_settings(&bnn_temp);

        auto bnn_memory_allocate = [BNN_LAMBDA_REFERENCE](
                bnn_bnn** bnn,
                bnn_bnn* bnn_settings
                ) -> void
        {
            if(!bnn_settings)
            {
                bnn_error_code = bnn_error_codes::error;
                return;
            }

            memory_.size = bnn_settings->memory_.size;
            printf("========================================== allocate_host_and_device_memory(memory_)\n");
            allocate_host_and_device_memory(memory_);

            if(!memory_.host_data || !memory_.device_data)
            {
                bnn_settings->bnn_error_code = bnn_error_codes::malloc_fail;
                return;
            }

            *bnn = reinterpret_cast<bnn_bnn*>(memory_.host_data);

            **bnn = *bnn_settings;
        };
        bnn_memory_allocate(&bnn_host, &bnn_temp);

        if(bnn_temp.bnn_error_code != bnn_error_codes::ok)
            return;

        bnn_calculate_pointers(bnn_host);
        bnn_fill_threads(bnn_host);

        for(u_word thread_number = 0; thread_number < bnn_host->threads_.size; ++thread_number)
        {
            bnn_fill_random_of_thread(bnn_host, thread_number);
            bnn_set_neurons_of_thread(bnn_host, thread_number);
        }

        int offset = (memory_.device_data - memory_.host_data) * sizeof(int);
        bnn_shift_pointers(bnn_host, offset);
        printf("========================================== gpu()\n");
    }

    void allocate_host_and_device_memory(memory& m)
    {
        checkCudaErrors(cudaMallocHost((void**)&m.host_data, m.size));
        memset(m.host_data, 0, m.size);

        checkCudaErrors(cudaMalloc((void**)&m.device_data, m.size));
        checkCudaErrors(cudaMemset(m.device_data, 0, m.size));

        int offset = (m.device_data - m.host_data) * sizeof(int);
        printf("********************* memory: host [%d], dev [%d], size [%d], delta [%d]\n",
               m.host_data, m.device_data, m.size, offset);
    }

    void free_host_and_device_memory()
    {
        checkCudaErrors(cudaFreeHost(memory_.host_data));
        checkCudaErrors(cudaFree(memory_.device_data));
        memory_.size = 0;
    }

    void memory_copy_host_to_device(memory& m)
    {
        //cudaMemcpyAsync(m.device_data, m.host_data, m.size, cudaMemcpyHostToDevice, 0);
        cudaMemcpy(m.device_data, m.host_data, m.size, cudaMemcpyHostToDevice);
    }

    void memory_copy_host_to_device()
    {
        memory_copy_host_to_device(memory_);
    }

    void memory_copy_device_to_host(memory& m)
    {
        //cudaMemcpyAsync(m.host_data, m.device_data, m.size, cudaMemcpyDeviceToHost, 0);
        cudaMemcpy(m.host_data, m.device_data, m.size, cudaMemcpyDeviceToHost);
    }

    void memory_copy_device_to_host()
    {
        memory_copy_device_to_host(memory_);
    }

    void start()
    {
        printf("========================================== start()\n");
        active = true;
        bnn_host->parameters_.stop = false;
        bnn_host->parameters_.start = true;
        thread = std::thread(run, this);
        thread.detach();
    }

    void stop()
    {
        printf("========================================== stop()\n");
        bnn_host->parameters_.start = false;
        bnn_host->parameters_.stop = true;
    }

    bool is_active()
    {
        return active;
    }

    static void run(gpu* me)
    {
        #include "undef_implementations.h"
        #include "../../bnn/bnn_implementation.h"

        // This will pick the best possible CUDA capable device
        const char** argv = nullptr;
        int devID = findCudaDevice(0, (const char **)argv);
        cudaDeviceProp cdp;
        checkCudaErrors(cudaGetDeviceProperties(&cdp, devID));

        // set kernel launch configuration
        dim3 threads = dim3(settings::calculate_thread_dim_x(cdp, me->bnn_host->threads_.size));
        printf("dim3 threads = %d %d\n", threads.x, me->bnn_host->threads_.size);
        dim3 blocks = dim3(settings::calculate_block_dim_x(cdp, me->bnn_host->threads_.size, threads));
        printf("dim3 blocks = %d\n", blocks.x);
        printf("threads count [%d]\n", threads.x * blocks.x);

        me->bnn_host->parameters_.start = true;

        printf("========================================== 001\n");
        me->memory_copy_host_to_device();

        for(int i = 0; i < 10; ++i)
        {
            printf("========================================== 002\n");
            increment_kernel<<<blocks, threads, 0, 0>>>(me->memory_.device_data, me->debug_memory.device_data);
            printf("========================================== 003\n");
            cudaMemcpy(me->debug_memory.host_data, me->debug_memory.device_data, me->debug_memory.size, cudaMemcpyDeviceToHost);
            printf("========================================== 004\n");
            for(u_word thread_number = 0; thread_number < me->bnn_host->threads_.size; ++thread_number)
                printf("========================================== 005 %d\n", me->debug_memory.host_data[thread_number]);
        }

        printf("========================================== 006\n");
        me->memory_copy_device_to_host();
        int offset = (me->memory_.device_data - me->memory_.host_data) * sizeof(int);
        bnn_shift_pointers(me->bnn_host, -offset);
        printf("========================================== 007\n");
        me->active = false;
    }

private:
    bool active{false};
    bnn_bnn* bnn_host{nullptr};
    memory memory_;
    memory debug_memory;
    std::thread thread;
};

} // namespace bnn::gpu::cuda

#endif // BNN_GPU_CUDA_GPU_HPP
