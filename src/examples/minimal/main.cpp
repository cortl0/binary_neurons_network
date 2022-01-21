/*
 *   Binary Neurons Network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#include "unistd.h"

#include <iostream>

#include "/usr/local/include/bnn/brain.h"

static const _word input_length = sizeof (int) * 8 - 1;
static const _word output_length = 8;
static char c[input_length + output_length + 32];
static bnn::brain brn(24, // random_array_length_in_power_of_two
                      16, // quantity_of_neurons_in_power_of_two
                      input_length,
                      output_length,
                      1 // threads_count_in_power_of_two (2^1=2)
                      );

static void cycle()
{
    int count = 0;
    c[count++] = 'i';
    c[count++] = 'n';
    c[count++] = '=';
    bool value;

    for (_word i = 0; i < input_length; i++)
    {
        // random numbers only demonstrates the workable of the algorithm
        // don't expect a wow effect without using real data
        value = rand()%2;

        c[count++] = static_cast<bool>(value) + 48;
        // Put data in the brain
        brn.set_input(i, value);
    }
    c[count++] = ' ';
    c[count++] = 'o';
    c[count++] = 'u';
    c[count++] = 't';
    c[count++] = '=';
    for (_word i = 0; i < output_length; i++)
    {
        // Get data from the brain
        c[count++] = brn.get_output(i) + 48;
    }
    c[count++] = '\0';
    std::cout << c << std::endl;
}

int main()
{
    brn.start();
    while(1)
    {
        usleep(100000);
        cycle();
    }

    return 0;
}
