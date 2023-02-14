/*
 *   Binary Neurons Network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#include "cpu.h"

#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <thread>

#include "bnn/bnn_implementation.h"
#include "common/logger.h"
#include "common/settings_converter.hpp"

namespace bnn
{
std::thread t;
//bnn_error_codes bnn_error_code = bnn_error_codes::ok;
//thread::thread()
//{
////    logging("");
//}

cpu::~cpu()
{
    free(bnn);
    logging("cpu::~cpu()");
}

cpu::cpu(const bnn_settings& bs)
{
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
            //bnn_error_code = bnn_error_codes::error;
            return;
        }

        void* memory = malloc(bnn_settings->memory_.size);

        if(!memory)
        {
            bnn_settings->bnn_error_code = bnn_error_codes::malloc_fail;
            return;
        }

        *bnn = reinterpret_cast<bnn_bnn*>(memory);

        **bnn = *bnn_settings;
    };
    bnn_memory_allocate(&bnn, &bnn_temp);

    if(bnn_temp.bnn_error_code != bnn_error_codes::ok)
        return;

    bnn_calculate_pointers(bnn);
    bnn_fill_threads(bnn);

    for(u_word thread_number = 0; thread_number < bnn->threads_.size; ++thread_number)
    {
        bnn_fill_random_of_thread(bnn, thread_number);
        bnn_set_neurons_of_thread(bnn, thread_number);
    }

    treads.resize(bnn_temp.threads_.size);

    logging("cpu::cpu()");
}

u_word thread_number = 0;
void cpu::start()
{
    bnn->parameters_.stop = false;

    for(u_word thread_number = 0; thread_number < bnn->threads_.size; ++thread_number)
    {
        treads[thread_number] = std::thread(bnn_thread_function, bnn, thread_number);
        treads[thread_number].detach();
    }

    main_tread = std::thread(bnn_bnn_function, bnn);
    main_tread.detach();
    bnn->parameters_.start = true;

    for(u_word i = 0; i < bnn->threads_.size; ++i)
        while(!bnn->threads_.data[i].in_work);
}

void cpu::stop()
{
    bnn->parameters_.start = false;
    bnn->parameters_.stop = true;

    for(u_word i = 0; i < bnn->threads_.size; ++i)
        while(bnn->threads_.data[i].in_work);
}

void cpu::set_input(u_word i, bool value)
{
    bnn->input_.data[i] = value;
}

bool cpu::get_output(u_word i)
{
    return bnn->output_.data[i];
}

bool cpu::is_active()
{
    // TODO
    return true;
}

} // namespace bnn
