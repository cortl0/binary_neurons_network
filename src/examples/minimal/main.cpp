/*
 *   Binary Neurons Network
 *   created by Ilya Shishkin
 *   cortl@yandex.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#include <iostream>
#include <thread>

#include "common/architecture.h"

int main()
{
    constexpr bnn_settings bs
    {
        .quantity_of_neurons_in_power_of_two = 12, // 2^12=4096
        .input_length = 31,
        .output_length = 8,
        .motor_binaries_per_motor = 8,
        .random_size_in_power_of_two = 22,
        .quantity_of_threads_in_power_of_two = 1, // 2^1=2
    };

    bnn::architecture bnn(bs);
    bnn.initialize();
    bnn.start();
    do {} while(bnn.get_state() != bnn_state::started);
    bool stop{false};

    auto manager_thread = std::thread([&stop]()
    {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        stop = true;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    });

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
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    bnn.stop();
    manager_thread.join();

    return 0;
}
