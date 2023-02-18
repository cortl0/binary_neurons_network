/*
 *   Binary Neurons Network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef BNN_UNIT_TESTS_COMMON_H
#define BNN_UNIT_TESTS_COMMON_H

#include <initializer_list>
#include <string_view>
#include "common/logger.h"

#define BNN_ARCHITECTURE_CPU

#define ASSERT_EQ(x, y)\
if((x) != (y))\
{\
    logging("error: expected ["s + std::to_string(x) + "], actual ["s + std::to_string(y) + "]"s);\
    return EXIT_FAILURE;\
}

namespace bnn::unit_tests
{

using namespace std::literals;

int launch_tests(const std::initializer_list<std::pair<int (*)(), std::string_view>> tests)
{
    for (auto& [test, name] : tests)
    {
        logging_test("begin      | "s + name.data());

        if(test())
            return EXIT_FAILURE;

        logging_test("successful | "s + name.data());
    }

    return EXIT_SUCCESS;
}

}
#endif // BNN_UNIT_TESTS_COMMON_H
