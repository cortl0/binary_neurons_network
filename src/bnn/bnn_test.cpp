#include <stdlib.h>
#include <cstdio>
#include <string_view>

#include "unit_tests/common.h"
#include "bnn/bnn_implementation.h"

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
        ASSERT_EQ(BNN_BYTES_ALIGNMENT * 2, size);
    }

    {
        u_word size{6};
        bnn_calculate_alignment(size);
        ASSERT_EQ(BNN_BYTES_ALIGNMENT * 2, size);
    }

    {
        u_word size{7};
        bnn_calculate_alignment(size);
        ASSERT_EQ(BNN_BYTES_ALIGNMENT * 2, size);
    }

    {
        u_word size{8};
        bnn_calculate_alignment(size);
        ASSERT_EQ(BNN_BYTES_ALIGNMENT * 2, size);
    }

    {
        u_word size{9};
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

//TODO
int bnn_set_neurons_of_thread_test()
{
    constexpr bnn_bnn good_bnn
    {
        .input_
        {
            .size = 3
        },
        .output_
        {
            .size = 7
        },
        .random_
        {
            .size_in_power_of_two = 8
        },
        .storage_
        {
            .size_in_power_of_two = 4
        },
        .motor_binaries_
        {
            .size_per_motor = 2
        },
        .threads_
        {
            .size_in_power_of_two = 1
        }
    };

    {
        bnn_bnn bnn_temp = good_bnn;
        ASSERT_EQ(bnn_error_codes::ok, bnn_calculate_settings(&bnn_temp));
        bnn_temp.memory_.data = malloc(bnn_temp.memory_.size);
        bnn_bnn* bnn = static_cast<bnn_bnn*>(bnn_temp.memory_.data);
        *bnn = bnn_temp;
        bnn_calculate_pointers(bnn);
        free(bnn);
    }

    auto prepare = [&]() -> bnn_bnn*
    {
        bnn_bnn bnn_temp = good_bnn;
        bnn_calculate_settings(&bnn_temp);
        bnn_temp.memory_.data = malloc(bnn_temp.memory_.size);
        bnn_bnn* bnn = static_cast<bnn_bnn*>(bnn_temp.memory_.data);
        *bnn = bnn_temp;
        bnn_calculate_pointers(bnn);
        return bnn;
    };

    {
        bnn_bnn* bnn = prepare();
        u_word size = 0;
        ASSERT_EQ((int64_t)bnn, (int64_t)bnn->memory_.data);
        bnn_calculate_alignment(size += sizeof(*bnn));
        ASSERT_EQ((int64_t)bnn, (int64_t)((char*)bnn->input_.data - size));
        bnn_calculate_alignment(size += bnn->input_.size);
        ASSERT_EQ((int64_t)bnn, (int64_t)((char*)bnn->output_.data - size));
        bnn_calculate_alignment(size += bnn->output_.size);
        ASSERT_EQ((int64_t)bnn, (int64_t)((char*)bnn->random_.data - size));
        bnn_calculate_alignment(size += bnn->random_.size * sizeof(*bnn->random_.data));
        ASSERT_EQ((int64_t)bnn, (int64_t)((char*)bnn->storage_.data - size));
        bnn_calculate_alignment(size += bnn->storage_.size * sizeof(*bnn->storage_.data));
        ASSERT_EQ((int64_t)bnn, (int64_t)((char*)bnn->motor_binaries_.data - size));
        bnn_calculate_alignment(size += bnn->motor_binaries_.size * sizeof(*bnn->motor_binaries_.data));
        ASSERT_EQ((int64_t)bnn, (int64_t)((char*)bnn->threads_.data - size));
        free(bnn);
    }

    {
        bnn_bnn* bnn = prepare();
        u_word size = 0;



        bnn_fill_threads(bnn);
        u_word thread_number = 0;
        bnn_set_neurons_of_thread(bnn, thread_number);
        thread_number = 1;
        bnn_set_neurons_of_thread(bnn, thread_number);


        ///*ASSERT_EQ(bnn_error_codes::ok,*/ bnn_set_neurons_of_thread(bnn, thread_number);//);


        free(bnn);
    }

    return EXIT_SUCCESS;
}

int main(int argc, char* argv[])
{
    const std::initializer_list<std::pair<int (*)(), std::string_view>> tests =
    {
        {bnn_calculate_alignment_test, "bnn_calculate_alignment_test"sv},
        {bnn_calculate_settings_test, "bnn_calculate_settings_test"sv},
        {bnn_set_neurons_of_thread_test, "bnn_set_neurons_of_thread_test"sv},
    };

    return bnn::unit_tests::launch_tests(tests);
}
