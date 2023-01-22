/*
 *   Binary Neurons Network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef BNN_COMMON_LOGGER_H
#define BNN_COMMON_LOGGER_H

#include <iostream>

#define get_time_def bnn::get_time()
#define place_string std::string().append(std::string(__FUNCTION__)).append(": ").append(__FILE__).append(": ").append(std::to_string(__LINE__))
#define log_string(msg) std::string().append(get_time_def).append(" | ").append(msg).append(" | at: ").append(place_string)
#define throw_error(msg) throw std::runtime_error(log_string(msg))
#define logging(msg) std::cout << log_string(msg) << std::endl

namespace bnn
{

std::string get_time();

} // namespace bnn

#endif // BNN_COMMON_LOGGER_H
