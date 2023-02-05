/*
 *   Binary Neurons Network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef BNN_COMMON_ARCHITECTURE_H
#define BNN_COMMON_ARCHITECTURE_H

#ifdef BNN_ARCHITECTURE_CPU
#include "cpu/cpu.h"
#elif defined BNN_ARCHITECTURE_CUDA
#include "gpu/cuda/cuda.h"
#endif

namespace bnn
{

#ifdef BNN_ARCHITECTURE_CPU
    using architecture = cpu;
#elif defined BNN_ARCHITECTURE_CUDA
    using architecture = gpu::cuda;
#endif

} // namespace bnn

#endif // BNN_COMMON_ARCHITECTURE_H
