/*
 *   binary neurons network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef BNN_GPU_CUDA_STATISTIC_H
#define BNN_GPU_CUDA_STATISTIC_H

#include <stdio.h>
#include <unistd.h>

#include "external/helper_cuda.h"

namespace bnn::gpu
{

class statistic
{
public:
    static void print_device_properties(cudaDeviceProp& cdp)
    {
      printf("CUDA device [%s]\n", cdp.name);
      printf("Total global mem:  %zu\n", cdp.totalGlobalMem);
      printf("Total constant Mem:  %zu\n", cdp.totalConstMem);
      printf("Total sharedMemPerBlock:  %zu\n\n", cdp.sharedMemPerBlock);
      printf("CUDA device multiProcessorCount [%d]\n", cdp.multiProcessorCount);
      printf("CUDA device maxBlocksPerMultiProcessor [%d]\n", cdp.maxBlocksPerMultiProcessor);
      printf("CUDA device maxThreadsPerMultiProcessor [%d]\n", cdp.maxThreadsPerMultiProcessor);
      printf("CUDA device maxThreadsPerBlock [%d]\n\n", cdp.maxThreadsPerBlock);
      printf("CUDA device maxThreadsDim [%d]\n", cdp.maxThreadsDim[0]);
      printf("CUDA device maxThreadsDim [%d]\n", cdp.maxThreadsDim[1]);
      printf("CUDA device maxThreadsDim [%d]\n\n", cdp.maxThreadsDim[2]);
      printf("CUDA device maxGridSize [%d]\n", cdp.maxGridSize[0]);
      printf("CUDA device maxGridSize [%d]\n", cdp.maxGridSize[1]);
      printf("CUDA device maxGridSize [%d]\n", cdp.maxGridSize[2]);
    }

    static void view_devices_list()
    {
      printf("===== Device list +++++\n");
      int deviceCount;
      cudaGetDeviceCount(&deviceCount);
      int device;
      for (device = 0; device < deviceCount; ++device)
      {
          cudaDeviceProp deviceProp;
          cudaGetDeviceProperties(&deviceProp, device);
          printf("Device %d has compute capability %d.%d.\n", device, deviceProp.major, deviceProp.minor);
          print_device_properties(deviceProp);
      }
      printf("===== Device list -----\n\n");
    }
};

} // namespace bnn::gpu

#endif // BNN_GPU_CUDA_STATISTIC_H
