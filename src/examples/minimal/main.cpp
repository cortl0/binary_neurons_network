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

#include "common/architecture.h"

int main()
{
    constexpr bnn_settings bs
    {
        .quantity_of_neurons_in_power_of_two = 12, // 2^12=4096
        .input_length = 31,
        .output_length = 8,
        .threads_count_in_power_of_two = 1 // 2^1=2
    };

    bnn::architecture bnn(bs);
    bnn.start();
    while(!bnn.is_active());
    bool stop{false};
    std::thread([&stop](){ sleep(1); stop = true; }).detach();

    while(!stop)
    {
        static char input[bs.input_length + 1]{};
        static char output[bs.output_length + 1]{};
        static bool value;

        for (u_word i = 0; i < bs.input_length; i++)
        {
            value = rand() % 2;

            // Put data in BNN
            bnn.set_input(i, value);

            input[i] = value + 48;
        }

        for (u_word i = 0; i < bs.output_length; i++)
        {
            // Get data from BNN
            value = bnn.get_output(i);

            output[i] = value + 48;
        }

        std::cout << "input=" << input << " output=" << output << std::endl;
        usleep(100000);
    }

    bnn.stop();

    return 0;
}
