/*
 *   Binary Neurons Network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

//#define BNN_ARCHITECTURE_CPU
#define BNN_ARCHITECTURE_CUDA

#include "bnn/bnn.h"

#ifdef BNN_ARCHITECTURE_CPU
#include "bnn/bnn_implementation.h"
#endif

#include <stdlib.h>
#include <iostream>

void foo()
{
    auto bnn_foo = [BNN_LAMBDA_REFERENCE](bnn_bnn* bnn_settings) -> void
    {
#ifdef BNN_ARCHITECTURE_CUDA
#include "bnn/bnn_implementation.h"
#endif

        auto bnn_memory_allocate = [BNN_LAMBDA_REFERENCE](
                bnn_bnn** bnn,
                bnn_bnn* bnn_settings
                ) -> void
        {
            if(!bnn_settings)
            {
                bnn_error_code = bnn_error_codes::ok;
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
        bnn_bnn* bnn{nullptr};
        bnn_calculate_settings(bnn_settings);
        bnn_memory_allocate(&bnn, bnn_settings);

        if(bnn_settings->bnn_error_code != bnn_error_codes::ok)
            return;
        bnn_calculate_pointers(bnn);
        bnn_fill_threads(bnn);

        for(u_word thread_number = 0; thread_number < bnn->threads_.size; ++thread_number)
        {
            bnn_fill_random_of_thread(bnn, thread_number);
            bnn_set_neurons_of_thread(bnn, thread_number);
        }

        bnn_shift_pointers(bnn, 0);

//        bnn->parameters_.start = false;
//        bnn->parameters_.stop = false;

//        for(u_word thread_number = 0; thread_number < bnn->threads_.size; ++thread_number)
//            bnn_thread_function(bnn, 0);

//        bnn_bnn_function(bnn);
//        bnn->parameters_.start = true;
    };

    bnn_bnn bnn_settings;
    bnn_settings.storage_.size_in_power_of_two = 16;
    bnn_settings.input_.size = 4;
    bnn_settings.output_.size = 2;
    bnn_settings.threads_.size_in_power_of_two = 1;
    bnn_foo(&bnn_settings);
}

int main()
{
    foo();
    printf("aaa\n");
    return 0;
}
