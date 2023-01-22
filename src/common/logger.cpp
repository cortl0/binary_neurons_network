/*
 *   Binary Neurons Network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#include "logger.h"

#include <ctime>
#include <thread>

namespace bnn
{

std::string get_time()
{
    std::string time_str;
#define LOGGER_TIME_BUFFER_LENGTH 20
#define LOGGER_TIME_BUFFER_MS_LENGTH 7
    char time_buffer[LOGGER_TIME_BUFFER_LENGTH];
    char time_buffer_ms[LOGGER_TIME_BUFFER_MS_LENGTH];
    std::time_t time;
    auto tp = std::chrono::high_resolution_clock::now().time_since_epoch();

    std::chrono::microseconds microseconds = std::chrono::duration_cast<std::chrono::microseconds>(tp);
    microseconds = microseconds % 1000000;
    std::sprintf(time_buffer_ms, "%.6d", static_cast<int>(microseconds.count()));

    std::chrono::seconds seconds = std::chrono::duration_cast<std::chrono::seconds>(tp);
    time = seconds.count();
    std::strftime(time_buffer, LOGGER_TIME_BUFFER_LENGTH, "%Y.%m.%d %H:%M:%S", std::localtime(&time));

    time_str.append(time_buffer).append(".").append(time_buffer_ms);

    return time_str;
};

} // namespace bnn
