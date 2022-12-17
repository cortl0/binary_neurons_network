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

#include "cpu/brain.h"

int main()
{
    int quantity_of_neurons_in_power_of_two = 12; // 2^12=4096
    int quantity_of_threads_in_power_of_two = 1; // 2^1=2
    const u_word input_length = 31;
    const u_word output_length = 8;
    char input[input_length + 1];
    char output[output_length + 1];
    input[input_length] = '\0';
    output[output_length] = '\0';
    bool value;

    bnn::brain brain_(
        quantity_of_neurons_in_power_of_two,
        input_length,
        output_length,
        quantity_of_threads_in_power_of_two
        );

    brain_.start();

    while(true)
    {
        for (u_word i = 0; i < input_length; i++)
        {
            value = rand() % 2;

            // Put data in bnn
            brain_.set_input(i, value);

            input[i] = value + 48;
        }

        for (u_word i = 0; i < output_length; i++)
        {
            // Get data from bnn
            value = brain_.get_output(i);

            output[i] = value + 48;
        }

        std::cout << "input=" << input << " output=" << output << std::endl;
        usleep(100000);
    }

    usleep(100000);
    brain_.stop();

    return 0;
}
