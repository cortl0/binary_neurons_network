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
