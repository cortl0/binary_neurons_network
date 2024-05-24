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

#include "external/helper_cuda.h"

namespace bnn::gpu
{

class settings
{
public:
    static int calculate_thread_dim_x(const cudaDeviceProp& cdp, int count)
    {
      if(count < 1)
        throw;

      if(count > cdp.maxThreadsPerMultiProcessor)
        count = cdp.maxThreadsPerMultiProcessor;

      if(count > cdp.maxThreadsPerBlock)
        count = cdp.maxThreadsPerBlock;

      if(count > cdp.maxThreadsDim[0])
        count = cdp.maxThreadsDim[0];

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

      return count;
    }
};

} // namespace bnn::gpu

#endif // BNN_GPU_CUDA_SETTINGS_H
