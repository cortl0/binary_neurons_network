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

#include "common/logger.h"
#include "settings.hpp"
#include "statistic.hpp"
#include "common/settings_converter.hpp"

#define BNN_CUDA_PRINT_PREFIX "======== "

__global__ void primary_filling(int* bnn_data, int* debug_data)
{
#include "undef_implementations.h"
#include "../../../bnn/bnn_implementation.h"

    int thread_number = blockIdx.x * blockDim.x + threadIdx.x;
    bnn_bnn* bnn = (bnn_bnn*)bnn_data;

    for(int i = 0; i < bnn->threads_.size; ++i)
        debug_data[i] = 0;

    bnn_fill_random_of_thread(bnn, thread_number);
    bnn_set_neurons_of_thread(bnn, thread_number);
    debug_data[thread_number] += thread_number + 1;
}

__global__ void start_cycle(int* bnn_data, int* debug_data)
{
#include "undef_implementations.h"
#include "../../../bnn/bnn_implementation.h"

    int thread_number = blockIdx.x * blockDim.x + threadIdx.x;
    bnn_bnn* bnn = (bnn_bnn*)bnn_data;
    bnn_thread_function(bnn, thread_number);
    debug_data[thread_number] += thread_number + 1;
}

namespace bnn
{

namespace gpu
{

cuda::~cuda()
{
    stop();
    while(active);

    if(cudaError::cudaSuccess != free_host_and_device_memory(memory_))
    {
        logging("fail: freeing memory_");
    }

    if(cudaError::cudaSuccess != free_host_and_device_memory(debug_memory))
    {
        logging("fail: freeing debug_memory");
    }

    logging("cuda::~cuda()");
}

cuda::cuda(const bnn_settings& bs)
{
#include "undef_implementations.h"
#include "../../bnn/bnn_implementation.h"

    debug_memory.size = 8192;

    if(!allocate_host_and_device_memory(debug_memory))
    {
        logging("fail: allocation debug_memory");
        throw EXIT_FAILURE;
    }

    bnn_bnn bnn_temp = convert_bnn_settings_to_bnn(bs);

    if(auto result = bnn_calculate_settings(&bnn_temp); bnn_error_codes::ok != result)
    {
        logging("bnn_error_code [" + std::to_string(result) + "]");
        throw static_cast<int>(result);
    }

    auto bnn_memory_allocate = [BNN_LAMBDA_REFERENCE](
            bnn_bnn** bnn,
            bnn_bnn* bnn_settings
            ) -> void
    {
        if(!bnn_settings)
        {
            logging("fail: bnn_settings is null");
            throw EXIT_FAILURE;
        }

        memory_.size = bnn_settings->memory_.size;

        if(!allocate_host_and_device_memory(memory_))
        {
            logging("fail: allocation memory_");
            throw EXIT_FAILURE;
        }

        *bnn = reinterpret_cast<bnn_bnn*>(memory_.host_data);

        **bnn = *bnn_settings;
    };

    bnn_memory_allocate(&bnn_host, &bnn_temp);
    bnn = bnn_host;
    bnn_calculate_pointers(bnn_host);
    bnn_fill_threads(bnn_host);
    logging("Success: cuda::cuda()");
}

bool cuda::get_output(u_word i)
{
    return bnn_host->output_.data[i];
}

void cuda::set_input(u_word i, bool value)
{
    bnn_host->input_.data[i] = value;
}

bool cuda::allocate_host_and_device_memory(memory& m)
{
    if(cudaError::cudaSuccess != cudaMallocHost((void**)&m.host_data, m.size))
    {
        logging("fail: allocation memory::host_data");
        return false;
    }

    if(cudaError::cudaSuccess != cudaMalloc((void**)&m.device_data, m.size))
    {
        logging("fail: allocation memory::device_data");
        cudaFreeHost(m.host_data);
        return false;
    }

    m.offset = (m.device_data - m.host_data) * sizeof(int);

    printf(BNN_CUDA_PRINT_PREFIX "memory: host [0x%016" PRIXPTR "], dev [0x%016" PRIXPTR "], size [%d], delta [%d]\n",
           (uintptr_t)m.host_data, (uintptr_t)m.device_data, m.size, m.offset);

    return true;
}

bool cuda::free_host_and_device_memory(memory& m)
{
    bool return_value{true};

    if(cudaError::cudaSuccess != cudaFree(m.device_data))
    {
        logging("fail: freeing memory::device_data");
        return_value = false;
    }

    if(cudaError::cudaSuccess != cudaFreeHost(m.host_data))
    {
        logging("fail: freeing memory::host_data");
        return_value = false;
    }

    m.size = 0;

    return return_value;
}

bool cuda::memory_copy_host_to_device(memory& m)
{
    //cudaMemcpyAsync(m.device_data, m.host_data, m.size, cudaMemcpyHostToDevice, 0);
    if(cudaError::cudaSuccess != cudaMemcpy(m.device_data, m.host_data, m.size, cudaMemcpyHostToDevice))
        return false;

    return true;
}

bool cuda::memory_copy_device_to_host(memory& m)
{
    //cudaMemcpyAsync(m.host_data, m.device_data, m.size, cudaMemcpyDeviceToHost, 0);
    if(cudaError::cudaSuccess != cudaMemcpy(m.host_data, m.device_data, m.size, cudaMemcpyDeviceToHost))
        return false;

    return true;
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

bool cuda::test_kernel_result()
{
    static u_word test[8192]{};

    bool return_value{true};

    for(u_word thread_number = 0; thread_number < bnn_host->threads_.size; ++thread_number)
    {
        test[thread_number] += thread_number + 1;

        if(debug_memory.host_data[thread_number] != test[thread_number])
            return_value = false;
    }

    return return_value;
}

void cuda::run(cuda* me)
{
    #include "undef_implementations.h"
    #include "../../bnn/bnn_implementation.h"

    const char** argv = nullptr;
    int devID = findCudaDevice(0, (const char **)argv);
    cudaDeviceProp cdp;
    checkCudaErrors(cudaGetDeviceProperties(&cdp, devID));
    statistic::view_devices_list();
    dim3 threads = dim3(settings::calculate_thread_dim_x(cdp, me->bnn_host->threads_.size));
    dim3 blocks = dim3(settings::calculate_block_dim_x(cdp, me->bnn_host->threads_.size, threads));
    printf("dim3 threads = %d\n", threads.x);
    printf("dim3 blocks = %d\n", blocks.x);
    printf("threads count [%d]\n", threads.x * blocks.x);

    me->bnn_host->parameters_.start = true;
    bnn_shift_pointers(me->bnn_host, me->memory_.offset);
    me->memory_copy_host_to_device(me->memory_);
    bnn_shift_pointers(me->bnn_host, -me->memory_.offset);
    primary_filling<<<blocks, threads, 0, 0>>>(me->memory_.device_data, me->debug_memory.device_data);

    printf(BNN_CUDA_PRINT_PREFIX "random_.size [%d]\n", me->bnn_host->random_.size);
    printf(BNN_CUDA_PRINT_PREFIX "grandom_.size_in_power_of_two [%d]\n", me->bnn_host->random_.size_in_power_of_two);

    if(!me->memory_copy_device_to_host(me->debug_memory))
        logging("fail: memory_copy_device_to_host(me->debug_memory) out while");

    if(!me->test_kernel_result())
    {
        printf(BNN_CUDA_PRINT_PREFIX "primary_filling is no good\n");
        me->active = true;
        sleep(2);
        me->active = false;
        return;
    }

    me->active = true;

    while(!me->bnn_host->parameters_.stop)
    {
        start_cycle<<<blocks, threads, 0, 0>>>(me->memory_.device_data, me->debug_memory.device_data);
        if(!me->memory_copy_device_to_host(me->debug_memory))
            logging("fail: memory_copy_device_to_host(me->debug_memory) in while");

        if(!me->test_kernel_result())
        {
            printf(BNN_CUDA_PRINT_PREFIX "start_cycle is no good\n");
            break;
        }

        checkCudaErrors(cudaMemcpy(me->bnn_host->output_.data, me->bnn_host->output_.data + me->memory_.offset,
                   me->bnn_host->output_.size, cudaMemcpyDeviceToHost));

        checkCudaErrors(cudaMemcpy(me->bnn_host->input_.data + me->memory_.offset, me->bnn_host->input_.data,
                   me->bnn_host->input_.size, cudaMemcpyHostToDevice));

        {
            bnn_bnn bnn_host_temp;
            checkCudaErrors(cudaMemcpy(&bnn_host_temp, (char*)me->bnn_host + me->memory_.offset,
                sizeof(bnn_host_temp), cudaMemcpyDeviceToHost));

            bool start_temp = me->bnn_host->parameters_.start;
            bool start_stop = me->bnn_host->parameters_.stop;

            me->bnn_host->parameters_ = bnn_host_temp.parameters_;
            me->bnn_host->debug_ = bnn_host_temp.debug_;

            me->bnn_host->parameters_.start = start_temp;
            me->bnn_host->parameters_.stop = start_stop;
        }
    }

//    checkCudaErrors(cudaMemcpy(&me->bnn_host->parameters_.stop + me->memory_.offset, &me->bnn_host->parameters_.stop,
//               sizeof(me->bnn_host->parameters_.stop), cudaMemcpyHostToDevice));

    printf(BNN_CUDA_PRINT_PREFIX "007\n");
    me->memory_copy_device_to_host(me->memory_);
    bnn_shift_pointers(me->bnn_host, -me->memory_.offset);
    printf(BNN_CUDA_PRINT_PREFIX "008\n");
    me->active = false;
}

} // namespace gpu

} // namespace bnn
