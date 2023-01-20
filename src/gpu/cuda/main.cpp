/*
 *   Binary Neurons Network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#include <stdio.h>
#include <unistd.h>

#include "statistic.hpp"
#include "cuda.hpp"

int main(int argc, char *argv[])
{
    printf("[%s] - Starting...\n", argv[0]);
    bnn::gpu::cuda::statistic::view_devices_list();

    bnn_settings bs;
    bs.quantity_of_neurons_in_power_of_two = 16; // 2^12=4096
    bs.threads_count_in_power_of_two = 2; // 2^1=2
    bs.input_length = 31;
    bs.output_length = 8;

    bnn::gpu::cuda::gpu cuda(bs);
    cuda.start();
//    sleep(3);
//    cuda.stop();
    //sleep(10);
    while(cuda.is_active());

    return 0;
}
