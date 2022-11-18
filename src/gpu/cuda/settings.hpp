/*
 *   binary neurons network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef BNN_GPU_CUDA_SETTINGS_H
#define BNN_GPU_CUDA_SETTINGS_H

#include <stdio.h>
#include <unistd.h>

#include "helper_cuda.h"

namespace bnn::gpu::cuda
{

class settings
{
public:
    static int calculate_thread_dim_x(const cudaDeviceProp& cdp, int count)
    {
      if(count < 1)
        throw;

      if(count > cdp.maxThreadsDim[0])
        count = cdp.maxThreadsDim[0];

      if(count > cdp.maxThreadsPerBlock)
        count = cdp.maxThreadsPerBlock;

      if(count > cdp.maxThreadsPerMultiProcessor)
        count = cdp.maxThreadsPerMultiProcessor;

      printf("threads_dim.x [%d]\n", count);

      return count;
    }

    static int calculate_block_dim_x(const cudaDeviceProp& cdp, int count, const dim3& threads)
    {
      if(count < 1)
        throw;

      if(count % threads.x)
        count = count / threads.x + 1;
      else
        count /= threads.x;

      printf("blocks_dim.x [%d]\n", count);

      return count;
    }
};

} // namespace bnn::gpu::cuda

#endif // BNN_GPU_CUDA_SETTINGS_H
