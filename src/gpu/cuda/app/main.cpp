/*
 *   Binary Neurons Network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#include "cuda.h"
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

int main()
{
    bnn_settings bs;
    bs.quantity_of_neurons_in_power_of_two = 16; // 2^12=4096
    bs.threads_count_in_power_of_two = 1; // 2^1=2
    bs.input_length = 4;
    bs.output_length = 2;

    bnn::gpu::cuda bnn(bs);
    bnn.start();
    sleep(1);
    bnn.stop();
    return 0;
}
