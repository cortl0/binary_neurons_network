/*
 *   Binary Neurons Network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#include "cuda.h"

#include <inttypes.h>
#include <stdlib.h>
#include <stdio.h>

#include "bnn/bnn_settings.h"
#include "settings.hpp"
#include "statistic.hpp"

#define BNN_CUDA_PRINT_PREFIX "======== "

__global__ void bnn_kernel(int* g_data, int* debug_data)
{
#include "undef_implementations.h"
#include "../../../bnn/bnn_implementation.h"

    int thread_number = blockIdx.x * blockDim.x + threadIdx.x;

    bnn_bnn* bnn = (bnn_bnn*)g_data;
    bnn_thread_function(bnn, thread_number);
    for(u_word thread_number = 0; thread_number < bnn->threads_.size; ++thread_number)
        debug_data[thread_number] = bnn->threads_.data[thread_number].iteration;
}

namespace bnn
{

namespace gpu
{

cuda::~cuda()
{
    stop();
    while(is_active());
    free_host_and_device_memory();
    printf(BNN_CUDA_PRINT_PREFIX "~gpu()\n");
}

cuda::cuda(const bnn_settings& bs)
{
#include "undef_implementations.h"
#include "../../bnn/bnn_implementation.h"

    debug_memory.size = 1024;
    printf(BNN_CUDA_PRINT_PREFIX "allocate_host_and_device_memory(debug_memory)\n");
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
            //bnn_error_code = bnn_error_codes::error;
            return;
        }

        memory_.size = bnn_settings->memory_.size;
        printf(BNN_CUDA_PRINT_PREFIX "allocate_host_and_device_memory(memory_)\n");
        allocate_host_and_device_memory(memory_);

        if(!memory_.host_data || !memory_.device_data)
        {
            printf(BNN_CUDA_PRINT_PREFIX "allocate_host_and_device_memory(memory_)\n");
            bnn_settings->bnn_error_code = bnn_error_codes::malloc_fail;
            return;
        }

        printf(BNN_CUDA_PRINT_PREFIX "allocate_host_and_device_memory(memory_)\n");

        *bnn = reinterpret_cast<bnn_bnn*>(memory_.host_data);

        **bnn = *bnn_settings;
    };
    bnn_memory_allocate(&bnn_host, &bnn_temp);

    bnn = bnn_host;

    if(bnn_temp.bnn_error_code != bnn_error_codes::ok)
        return;

    bnn_calculate_pointers(bnn_host);
    bnn_fill_threads(bnn_host);

    for(u_word thread_number = 0; thread_number < bnn_host->threads_.size; ++thread_number)
    {
        bnn_fill_random_of_thread(bnn_host, thread_number);
        bnn_set_neurons_of_thread(bnn_host, thread_number);
    }

    printf(BNN_CUDA_PRINT_PREFIX "fin gpu()\n");
}

bool cuda::get_output(u_word i)
{
    return bnn_host->output_.data[i];
}

void cuda::set_input(u_word i, bool value)
{
    bnn_host->input_.data[i] = value;
}

void cuda::allocate_host_and_device_memory(memory& m)
{
    checkCudaErrors(cudaMallocHost((void**)&m.host_data, m.size));
    memset(m.host_data, 0, m.size);
    checkCudaErrors(cudaMalloc((void**)&m.device_data, m.size));
    checkCudaErrors(cudaMemset(m.device_data, 0, m.size));
    m.offset = (m.device_data - m.host_data) * sizeof(int);

    printf(BNN_CUDA_PRINT_PREFIX "memory: host [0x%016" PRIXPTR "], dev [0x%016" PRIXPTR "], size [%d], delta [%d]\n",
           (uintptr_t)m.host_data, (uintptr_t)m.device_data, m.size, m.offset);
}

void cuda::free_host_and_device_memory()
{
    checkCudaErrors(cudaFreeHost(memory_.host_data));
    checkCudaErrors(cudaFree(memory_.device_data));
    memory_.size = 0;
}

void cuda::memory_copy_host_to_device(memory& m)
{
    //cudaMemcpyAsync(m.device_data, m.host_data, m.size, cudaMemcpyHostToDevice, 0);
    checkCudaErrors(cudaMemcpy(m.device_data, m.host_data, m.size, cudaMemcpyHostToDevice));
}

void cuda::memory_copy_host_to_device()
{
    memory_copy_host_to_device(memory_);
}

void cuda::memory_copy_device_to_host(memory& m)
{
    //cudaMemcpyAsync(m.host_data, m.device_data, m.size, cudaMemcpyDeviceToHost, 0);
    checkCudaErrors(cudaMemcpy(m.host_data, m.device_data, m.size, cudaMemcpyDeviceToHost));
}

void cuda::memory_copy_device_to_host()
{
    memory_copy_device_to_host(memory_);
}

void cuda::start()
{
    printf(BNN_CUDA_PRINT_PREFIX "start()\n");
    bnn_host->parameters_.stop = false;
    bnn_host->parameters_.start = true;
    thread = std::thread(run, this);
    thread.detach();
    while(!active);
}

void cuda::stop()
{
    printf(BNN_CUDA_PRINT_PREFIX "stop()\n");
    bnn_host->parameters_.start = false;
    bnn_host->parameters_.stop = true;
    while(active);
}

bool cuda::is_active()
{
    return active;
}

void cuda::run(cuda* me)
{
    #include "undef_implementations.h"
    #include "../../bnn/bnn_implementation.h"

    // This will pick the best possible CUDA capable device
    const char** argv = nullptr;
    int devID = findCudaDevice(0, (const char **)argv);
    cudaDeviceProp cdp;
    checkCudaErrors(cudaGetDeviceProperties(&cdp, devID));

//    // set kernel launch configuration
    dim3 threads = dim3(settings::calculate_thread_dim_x(cdp, me->bnn_host->threads_.size));
//    printf("dim3 threads = %d %d\n", threads.x, me->bnn_host->threads_.size);
    dim3 blocks = dim3(settings::calculate_block_dim_x(cdp, me->bnn_host->threads_.size, threads));
//    printf("dim3 blocks = %d\n", blocks.x);
//    printf("threads count [%d]\n", threads.x * blocks.x);

    me->bnn_host->parameters_.start = true;

    printf(BNN_CUDA_PRINT_PREFIX "001\n");
    bnn_shift_pointers(me->bnn_host, me->memory_.offset);
    printf(BNN_CUDA_PRINT_PREFIX "002\n");
    me->memory_copy_host_to_device();
    printf(BNN_CUDA_PRINT_PREFIX "003\n");
    bnn_shift_pointers(me->bnn_host, -me->memory_.offset);
    printf(BNN_CUDA_PRINT_PREFIX "004\n");

    me->active = true;
    while(!me->bnn_host->parameters_.stop)
    {
        bnn_kernel<<<blocks, threads, 0, 0>>>(me->memory_.device_data, me->debug_memory.device_data);

        checkCudaErrors(cudaMemcpy(me->debug_memory.host_data, me->debug_memory.device_data,
                   me->debug_memory.size, cudaMemcpyDeviceToHost));

        checkCudaErrors(cudaMemcpy(me->bnn_host->output_.data, me->bnn_host->output_.data + me->memory_.offset,
                   me->bnn_host->output_.size, cudaMemcpyDeviceToHost));

        checkCudaErrors(cudaMemcpy(me->bnn_host->input_.data + me->memory_.offset, me->bnn_host->input_.data,
                   me->bnn_host->input_.size, cudaMemcpyHostToDevice));

//        for(u_word thread_number = 0; thread_number < me->bnn_host->threads_.size; ++thread_number)
//            printf("========================================== 005 %d\n", me->debug_memory.host_data[thread_number]);
    }

//    checkCudaErrors(cudaMemcpy(&me->bnn_host->parameters_.stop + me->memory_.offset, &me->bnn_host->parameters_.stop,
//               sizeof(me->bnn_host->parameters_.stop), cudaMemcpyHostToDevice));

    //todo while

    printf(BNN_CUDA_PRINT_PREFIX "006\n");
    me->memory_copy_device_to_host();
    bnn_shift_pointers(me->bnn_host, -me->memory_.offset);
    printf(BNN_CUDA_PRINT_PREFIX "007\n");
    me->active = false;
}

} // namespace gpu

} // namespace bnn
