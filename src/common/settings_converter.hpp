/*
 *   Binary Neurons Network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef BNN_COMMON_SETTINGS_CONVERTER_H
#define BNN_COMMON_SETTINGS_CONVERTER_H

#include "bnn/bnn.h"
#include "common/settings.h"

namespace bnn
{

inline bnn_bnn convert_bnn_settings_to_bnn(const bnn_settings& bs)
{
    bnn_bnn bnn;
    bnn.storage_.size_in_power_of_two = bs.quantity_of_neurons_in_power_of_two;
    bnn.input_.size = bs.input_length;
    bnn.output_.size = bs.output_length;
    bnn.motor_binaries_.size_per_motor = bs.motor_binaries_per_motor;
    bnn.random_.size_in_power_of_two = bs.random_size_in_power_of_two;
    bnn.threads_.size_in_power_of_two = bs.quantity_of_threads_in_power_of_two;
    return bnn;
}

} // namespace bnn

#endif // BNN_COMMON_SETTINGS_CONVERTER_H
