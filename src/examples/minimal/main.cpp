/*
 *   Binary Neurons Network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#include <unistd.h>

#include <iostream>

#ifdef BNN_ARCHITECTURE_CPU
#include "cpu/cpu.h"
#endif

#ifdef BNN_ARCHITECTURE_CUDA
#include "gpu/cuda/cuda.h"
#endif

int main()
{
    bool value;
    bnn_settings bs;
    bs.quantity_of_neurons_in_power_of_two = 12; // 2^12=4096
    bs.threads_count_in_power_of_two = 1; // 2^1=2
    bs.input_length = 31;
    bs.output_length = 8;
    char input[bs.input_length + 1];
    char output[bs.output_length + 1];
    input[bs.input_length] = '\0';
    output[bs.output_length] = '\0';

#ifdef BNN_ARCHITECTURE_CPU
    bnn::cpu brain_(bs);
#endif

#ifdef BNN_ARCHITECTURE_CUDA
    bnn::gpu::cuda brain_(bs);
#endif

    brain_.start();
    while(!brain_.is_active());
    bool stop{false};
    std::thread([&](){ sleep(1); stop = true; }).detach();

    while(!stop)
    {
        for (u_word i = 0; i < bs.input_length; i++)
        {
            value = rand() % 2;

            // Put data in bnn
            brain_.set_input(i, value);

            input[i] = value + 48;
        }

        for (u_word i = 0; i < bs.output_length; i++)
        {
            // Get data from bnn
            value = brain_.get_output(i);

            output[i] = value + 48;
        }

        std::cout << "input=" << input << " output=" << output << std::endl;
        usleep(100000);
    }

    brain_.stop();

    return 0;
}
