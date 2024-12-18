/*
 *   Binary Neurons Network
 *   created by Ilya Shishkin
 *   cortl@yandex.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#include "cpu.h"


#include "bnn/bnn_implementation.h"
#include "common/logger.h"
#include "common/settings_converter.hpp"

namespace bnn
{

//bnn_error_codes bnn_error_code = bnn_error_codes::ok;

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

        void* memory = malloc(bnn_settings->parameters_.size);

        if(!memory)
        {
            bnn_settings->parameters_.bnn_error_code = bnn_error_codes::malloc_fail;
            return;
        }

        *bnn = reinterpret_cast<bnn_bnn*>(memory);

        **bnn = *bnn_settings;
    };

    bnn_memory_allocate(&bnn, &bnn_temp);

    if(bnn_temp.parameters_.bnn_error_code != bnn_error_codes::ok)
        return;

    bnn_calculate_pointers(bnn);
    bnn_fill_threads(bnn);

    for(u_word thread_number = 0; thread_number < bnn->threads_.size; ++thread_number)
    {
        bnn_fill_random_of_thread(bnn, thread_number);
        bnn_set_neurons_of_thread(bnn, thread_number);
        bnn_create_fake_binary_neurons_of_thread(bnn, thread_number);
    }

    treads.resize(bnn_temp.threads_.size);

    logging("cpu::cpu()");
}

void cpu::calculate_pointers()
{
    bnn_calculate_pointers(bnn);
}

u_word thread_number = 0;

void cpu::start()
{
    if(bnn->parameters_.state != bnn_state::stopped)
        return;

    bnn->parameters_.state = bnn_state::start;

    for(u_word thread_number = 0; thread_number < bnn->threads_.size; ++thread_number)
    {
        treads[thread_number] = std::thread(bnn_thread_function, bnn, thread_number);
        treads[thread_number].detach();
    }

    main_thread = std::thread(bnn_bnn_function, bnn);
    main_thread.detach();
    bnn->parameters_.state = bnn_state::started;

    for(u_word i = 0; i < bnn->threads_.size; ++i)
        while(!bnn->threads_.data[i].in_work);
}

void cpu::stop()
{
    if(bnn->parameters_.state != bnn_state::started)
        return;

    bnn->parameters_.state = bnn_state::stop;

    for(u_word i = 0; i < bnn->threads_.size; ++i)
        while(bnn->threads_.data[i].in_work);

    bnn->parameters_.state = bnn_state::stopped;
}

void cpu::set_input(u_word i, bool value)
{
    bnn->input_.data[i] = value;
}

bool cpu::get_output(u_word i)
{
    return bnn->output_.data[i];
}

bnn_state cpu::get_state()
{
    return bnn->parameters_.state;
}

} // namespace bnn
