/*
 *   Binary Neurons Network
 *   created by Ilya Shishkin
 *   cortl@yandex.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#include "cuda.h"

#include <inttypes.h>

#include <chrono>
#include <thread>

#include "common/logger.h"
#include "settings.hpp"
#include "statistic.hpp"
#include "common/settings_converter.hpp"

#define BNN_CUDA_PRINT_PREFIX "======== "

__global__ void primary_filling(int* bnn_data, int* debug_data)
{
#include "undef_implementations.h"
#include "bnn/bnn_implementation.h"

    int thread_number = blockIdx.x * blockDim.x + threadIdx.x;
    bnn_bnn* bnn = (bnn_bnn*)bnn_data;

    for(int i = 0; i < bnn->threads_.size; ++i)
        debug_data[i] = 0;

    bnn_fill_random_of_thread(bnn, thread_number);

//    bnn_random_corret_sum(
//            &bnn->random_,
//            bnn->threads_.data[thread_number].random_config.debug_.random_.sum_put,
//            &bnn->threads_.data[thread_number].random_config);

    bnn_set_neurons_of_thread(bnn, thread_number);
    bnn_create_fake_binary_neurons_of_thread(bnn, thread_number);

    debug_data[thread_number] += thread_number + 1;
}

__global__ void start_cycle(int* bnn_data, int* debug_data)
{
#include "undef_implementations.h"
#include "bnn/bnn_implementation.h"

    int thread_number = blockIdx.x * blockDim.x + threadIdx.x;
    bnn_bnn* bnn = (bnn_bnn*)bnn_data;
    bnn_thread_function(bnn, thread_number);
    debug_data[thread_number] += thread_number + 1;
}

namespace bnn
{

namespace gpu
{

using namespace std::chrono_literals;

cuda::~cuda()
{
    stop();

    one_time_trigger = false;

    while(!one_time_trigger_stop)
        std::this_thread::sleep_for(1ms);

    if(cudaError::cudaSuccess != free_host_and_device_memory(memory_))
        logging("fail: freeing memory_");

    if(cudaError::cudaSuccess != free_host_and_device_memory(debug_memory))
        logging("fail: freeing debug_memory");

    logging("cuda::~cuda()");
}

cuda::cuda(const bnn_settings& bs)
{
#include "undef_implementations.h"
#include "bnn/bnn_implementation.h"

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

        memory_.size = bnn_settings->parameters_.size;

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

void cuda::calculate_pointers()
{
#include "undef_implementations.h"
#include "bnn/bnn_implementation.h"
    bnn_calculate_pointers(bnn_host);
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

void cuda::initialize()
{
    std::thread(&cuda::run, this).detach();
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
    printf(BNN_CUDA_PRINT_PREFIX "start(begin)\n");

    if(state != bnn_state::stopped)
        return;

    state = bnn_state::start;

    while(state == bnn_state::start)
        std::this_thread::sleep_for(1ms);

    printf(BNN_CUDA_PRINT_PREFIX "start(end)==\n");
}

void cuda::stop()
{
    printf(BNN_CUDA_PRINT_PREFIX "stop(begin)\n");

    if(state != bnn_state::started)
        return;

    state = bnn_state::stop;

    while(state == bnn_state::stop)
        std::this_thread::sleep_for(1ms);

    printf(BNN_CUDA_PRINT_PREFIX "stop(end)\n");
}

bnn_state cuda::get_state()
{
    return state;
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

void cuda::run()
{
    #include "undef_implementations.h"
    #include "bnn/bnn_implementation.h"
    const char** argv = nullptr;
    int devID = findCudaDevice(0, (const char **)argv);
    cudaDeviceProp cdp;
    checkCudaErrors(cudaGetDeviceProperties(&cdp, devID));
    statistic::view_devices_list();
    dim3 threads = dim3(settings::calculate_thread_dim_x(cdp, bnn_host->threads_.size));
    dim3 blocks = dim3(settings::calculate_block_dim_x(cdp, bnn_host->threads_.size, threads));
    printf("dim3 threads = %d\n", threads.x);
    printf("dim3 blocks = %d\n", blocks.x);
    printf("threads count [%d]\n", threads.x * blocks.x);
    bnn_host->parameters_.state = bnn_state::started;
    bnn_shift_pointers(bnn_host, memory_.offset);
    memory_copy_host_to_device(memory_);
    bnn_shift_pointers(bnn_host, -memory_.offset);
    primary_filling<<<blocks, threads, 0, 0>>>(memory_.device_data, debug_memory.device_data);
    printf(BNN_CUDA_PRINT_PREFIX "random_.size [%d]\n", bnn_host->random_.size);
    printf(BNN_CUDA_PRINT_PREFIX "random_.size_in_power_of_two [%d]\n", bnn_host->random_.size_in_power_of_two);

    if(!memory_copy_device_to_host(debug_memory))
        logging("fail: memory_copy_device_to_host(debug_memory) out while");

    if(!test_kernel_result())
    {
        printf(BNN_CUDA_PRINT_PREFIX "primary_filling is no good\n");
        state = bnn_state::started;
        std::this_thread::sleep_for(2s);
        state = bnn_state::stopped;
        return;
    }

    state = bnn_state::started;

    while(one_time_trigger)
    {
        if(state == bnn_state::stop)
            state = bnn_state::stopped;

        if(state == bnn_state::start)
            state = bnn_state::started;

        if(bnn_state::started != state)
        {
            std::this_thread::sleep_for(1ms);
            continue;
        }

        start_cycle<<<blocks, threads, 0, 0>>>(memory_.device_data, debug_memory.device_data);
        if(!memory_copy_device_to_host(debug_memory))
            logging("fail: memory_copy_device_to_host(debug_memory) in while");

        if(!test_kernel_result())
        {
            printf(BNN_CUDA_PRINT_PREFIX "start_cycle is no good\n");
            break;
        }

        checkCudaErrors(cudaMemcpy(bnn_host->output_.data, bnn_host->output_.data + memory_.offset,
                   bnn_host->output_.size, cudaMemcpyDeviceToHost));

        checkCudaErrors(cudaMemcpy(bnn_host->input_.data + memory_.offset, bnn_host->input_.data,
                   bnn_host->input_.size, cudaMemcpyHostToDevice));

        {
            bnn_bnn bnn_host_temp;
            checkCudaErrors(cudaMemcpy(&bnn_host_temp, (char*)bnn_host + memory_.offset,
                sizeof(bnn_host_temp), cudaMemcpyDeviceToHost));

            bnn_host->parameters_ = bnn_host_temp.parameters_;
            bnn_host->debug_ = bnn_host_temp.debug_;
        }
    }

    bnn_host->parameters_.state = bnn_state::stop;
    checkCudaErrors(cudaMemcpy((char*)&bnn_host->parameters_.state + memory_.offset, &bnn_host->parameters_.state,
            sizeof(bnn_host->parameters_.state), cudaMemcpyHostToDevice));

    while(true)
    {
        bnn_bnn bnn_host_temp;

        checkCudaErrors(cudaMemcpy(&bnn_host_temp, (char*)bnn_host + memory_.offset,
            sizeof(bnn_host_temp), cudaMemcpyDeviceToHost));

        bnn_shift_pointers(&bnn_host_temp, -memory_.offset);
        bool result{};

        for(u_word i = 0; i < bnn_host_temp.threads_.size; ++i)
            if(bnn_host_temp.threads_.data[i].in_work)
                result = true;

        if(!result)
            break;
    }

    bnn_host->parameters_.state = bnn_state::stopped;
    state = bnn_state::stopped;
    one_time_trigger_stop = true;
}

void cuda::upload()
{
    #include "undef_implementations.h"
    #include "bnn/bnn_implementation.h"
    bnn_shift_pointers(bnn_host, memory_.offset);
    memory_copy_host_to_device(memory_);
    bnn_shift_pointers(bnn_host, -memory_.offset);
}

void cuda::download()
{
    #include "undef_implementations.h"
    #include "bnn/bnn_implementation.h"
    memory_copy_device_to_host(memory_);
    bnn_shift_pointers(bnn_host, -memory_.offset);
}

} // namespace gpu

} // namespace bnn
