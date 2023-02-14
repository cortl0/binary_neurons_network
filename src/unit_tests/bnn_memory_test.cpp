#include <stdlib.h>
#include <cstdio>

#define BNN_ARCHITECTURE_CPU

#include "bnn/bnn_implementation.h"
#include "common/logger.h"

#define ASSERT_EQ(x, y)\
if((x) != (y))\
{\
    logging("error: expected ["s + std::to_string(x) + "], actual ["s + std::to_string(y) + "]"s);\
    return EXIT_FAILURE;\
}

using namespace std::literals;

int bnn_calculate_alignment_test()
{
    {
        u_word size{0};
        bnn_calculate_alignment(size);
        ASSERT_EQ(0, size);
    }

    {
        u_word size{1};
        bnn_calculate_alignment(size);
        ASSERT_EQ(BNN_BYTES_ALIGNMENT, size);
    }

    {
        u_word size{2};
        bnn_calculate_alignment(size);
        ASSERT_EQ(BNN_BYTES_ALIGNMENT, size);
    }

    {
        u_word size{3};
        bnn_calculate_alignment(size);
        ASSERT_EQ(BNN_BYTES_ALIGNMENT, size);
    }

    {
        u_word size{4};
        bnn_calculate_alignment(size);
        ASSERT_EQ(BNN_BYTES_ALIGNMENT, size);
    }

    {
        u_word size{5};
        bnn_calculate_alignment(size);
        ASSERT_EQ(BNN_BYTES_ALIGNMENT, size);
    }

    {
        u_word size{6};
        bnn_calculate_alignment(size);
        ASSERT_EQ(BNN_BYTES_ALIGNMENT, size);
    }

    {
        u_word size{7};
        bnn_calculate_alignment(size);
        ASSERT_EQ(BNN_BYTES_ALIGNMENT, size);
    }

    {
        u_word size{8};
        bnn_calculate_alignment(size);
        ASSERT_EQ(BNN_BYTES_ALIGNMENT, size);
    }

    {
        u_word size{9};
        bnn_calculate_alignment(size);
        ASSERT_EQ(BNN_BYTES_ALIGNMENT * 2, size);
    }

    {
        u_word size{10};
        bnn_calculate_alignment(size);
        ASSERT_EQ(BNN_BYTES_ALIGNMENT * 2, size);
    }

    {
        u_word size{11};
        bnn_calculate_alignment(size);
        ASSERT_EQ(BNN_BYTES_ALIGNMENT * 2, size);
    }

    {
        u_word size{12};
        bnn_calculate_alignment(size);
        ASSERT_EQ(BNN_BYTES_ALIGNMENT * 2, size);
    }

    {
        u_word size{13};
        bnn_calculate_alignment(size);
        ASSERT_EQ(BNN_BYTES_ALIGNMENT * 2, size);
    }

    {
        u_word size{14};
        bnn_calculate_alignment(size);
        ASSERT_EQ(BNN_BYTES_ALIGNMENT * 2, size);
    }

    {
        u_word size{15};
        bnn_calculate_alignment(size);
        ASSERT_EQ(BNN_BYTES_ALIGNMENT * 2, size);
    }

    {
        u_word size{16};
        bnn_calculate_alignment(size);
        ASSERT_EQ(BNN_BYTES_ALIGNMENT * 2, size);
    }

    {
        u_word size{17};
        bnn_calculate_alignment(size);
        ASSERT_EQ(BNN_BYTES_ALIGNMENT * 3, size);
    }

    return EXIT_SUCCESS;
}

int bnn_calculate_settings_test()
{
    constexpr bnn_bnn good_bnn
    {
        .input_
        {
            .size = 3
        },
        .output_
        {
            .size = 5
        },
        .random_
        {
            .size_in_power_of_two = 21
        },
        .storage_
        {
            .size_in_power_of_two = 12
        },
        .motor_binaries_
        {
            .size_per_motor = 1
        },
        .threads_
        {
            .size_in_power_of_two = 1
        }
    };

    {
        bnn_bnn bnn = good_bnn;
        ASSERT_EQ(bnn_error_codes::ok, bnn_calculate_settings(&bnn));
        ASSERT_EQ(bnn.motor_binaries_.size_per_motor * bnn.output_.size, bnn.motor_binaries_.size);

        ASSERT_EQ((1 << bnn.random_.size_in_power_of_two) / QUANTITY_OF_BITS_IN_WORD, bnn.random_.size);
        ASSERT_EQ(1 << bnn.storage_.size_in_power_of_two, bnn.storage_.size);
        ASSERT_EQ(1 << bnn.threads_.size_in_power_of_two, bnn.threads_.size);
        ASSERT_EQ((1 << bnn.storage_.size_in_power_of_two) / (1 << bnn.threads_.size_in_power_of_two), bnn.threads_.neurons_per_thread);
        ASSERT_EQ(bnn.parameters_.random_config.put_offset_end, bnn.random_.size);

        u_word size = 0;
        size += sizeof(bnn_bnn);
        bnn_calculate_alignment(size);
        size += sizeof(bool) * bnn.input_.size;
        bnn_calculate_alignment(size);
        size += sizeof(bool) * bnn.output_.size;
        bnn_calculate_alignment(size);
        size += sizeof(u_word) * bnn.random_.size;
        bnn_calculate_alignment(size);
        size += sizeof(bnn_storage) * bnn.storage_.size;
        bnn_calculate_alignment(size);
        size += sizeof(bnn_motor::binary) * bnn.motor_binaries_.size;
        bnn_calculate_alignment(size);
        size += sizeof(bnn_thread) * bnn.threads_.size;
        bnn_calculate_alignment(size);
        ASSERT_EQ(size, bnn.memory_.size);
    }

    {
        bnn_bnn bnn = good_bnn;
        bnn.input_.size = 0;
        ASSERT_EQ(bnn_error_codes::input_size_must_be_greater_than_zero, bnn_calculate_settings(&bnn));
    }

    {
        bnn_bnn bnn = good_bnn;
        bnn.output_.size = 0;
        ASSERT_EQ(bnn_error_codes::output_size_must_be_greater_than_zero, bnn_calculate_settings(&bnn));
    }

    {
        bnn_bnn bnn = good_bnn;
        bnn.motor_binaries_.size_per_motor = 0;
        ASSERT_EQ(bnn_error_codes::motor_binaries_size_per_motor_must_be_greater_than_zero, bnn_calculate_settings(&bnn));
    }

    {
        bnn_bnn bnn = good_bnn;
        bnn.storage_.size_in_power_of_two = 0;
        ASSERT_EQ(bnn_error_codes::storage_size_too_small, bnn_calculate_settings(&bnn));
    }

    {
        bnn_bnn bnn = good_bnn;
        bnn.threads_.size_in_power_of_two = 13;
        ASSERT_EQ(bnn_error_codes::neurons_per_thread_must_be_greater_than_zero, bnn_calculate_settings(&bnn));
    }

    {
        bnn_bnn bnn = good_bnn;
        bnn.random_.size_in_power_of_two = QUANTITY_OF_BITS_IN_WORD;
        ASSERT_EQ(bnn_error_codes::random_size_in_power_of_two_must_be_less_then_quantity_of_bits_in_word, bnn_calculate_settings(&bnn));
    }

    return EXIT_SUCCESS;
}

int main()
{
    logging("bnn_calculate_alignment_test() begin"s);
    if(bnn_calculate_alignment_test())
        return EXIT_FAILURE;
    logging("bnn_calculate_alignment_test() success"s);

    logging("bnn_calculate_settings_test() begin"s);
    if(bnn_calculate_settings_test())
        return EXIT_FAILURE;
    logging("bnn_calculate_settings_test() success"s);

    return EXIT_SUCCESS;
}
