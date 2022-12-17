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

    int n = 1 << 20;
    bnn::gpu::cuda::memory_.size = n * sizeof(int);
    int value = 1 << 23;

    bnn::gpu::cuda::gpu::run(bnn::gpu::cuda::memory_, value, n);

    return 0;
}
