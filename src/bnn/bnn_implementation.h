/*
 *   Binary Neurons Network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef BNN_BNN_IMPLEMENTATION_H
#define BNN_BNN_IMPLEMENTATION_H

//#include "neurons/storage_implementation.h"
#include "thread_implementation.h"

auto bnn_bnn_set = [](
        bnn_bnn** bnn_,
        u_word quantity_of_neurons_in_power_of_two,
        u_word input_length,
        u_word output_length,
        u_word threads_count_in_power_of_two = 0
        ) -> void
{
    const u_word quantity_of_motor_binaries_per_motor = 16;
    const u_word quantity_of_motor_binaries = quantity_of_motor_binaries_per_motor * output_length;
    u_word quantity_of_neurons = bnn_math_two_pow_x(quantity_of_neurons_in_power_of_two);
    u_word threads_size = bnn_math_two_pow_x(threads_count_in_power_of_two);
    u_word quantity_of_neurons_binary = quantity_of_neurons - input_length - output_length;

    if(quantity_of_neurons <= input_length + output_length)
    {
        //throw_error("quantity_of_neurons_sensor + quantity_of_neurons_motor >= quantity_of_neurons_end");
        exit(1);
    }

    u_word random_array_length_in_power_of_two;

    auto calculate_random_size_bits = [&]() -> u_word
    {
        const u_word experimental_coefficient = bnn_math_log2_1(QUANTITY_OF_BITS_IN_WORD) + 5;
        random_array_length_in_power_of_two = quantity_of_neurons_in_power_of_two + experimental_coefficient;

        if(random_array_length_in_power_of_two >= QUANTITY_OF_BITS_IN_WORD)
        {
            random_array_length_in_power_of_two = QUANTITY_OF_BITS_IN_WORD - 1;
            //logging("random_array_length_in_power_of_two does not satisfy of experimental_coefficient");
        }

        if(random_array_length_in_power_of_two > 29)
        {
            random_array_length_in_power_of_two = 29;
            //logging("random_array_length_in_power_of_two hardware limitation");
        }

        return (1 << random_array_length_in_power_of_two) / QUANTITY_OF_BITS_IN_WORD;
    };
    u_word random_size_words = calculate_random_size_bits();

    u_word memory_size{0};
    memory_size += sizeof(bnn_bnn);
    memory_size += sizeof(bool) * input_length;
    memory_size += sizeof(bool) * output_length;
    memory_size += sizeof(u_word) * random_size_words;
    memory_size += sizeof(bnn_storage) * quantity_of_neurons;
    memory_size += sizeof(bnn_motor::binary) * quantity_of_motor_binaries;
    memory_size += sizeof(bnn_thread) * threads_size;

    void* memory = malloc(memory_size);

    if(!memory)
        exit(1);

    bnn_bnn* bnn = reinterpret_cast<bnn_bnn*>(memory);

    bnn->memory_.size = memory_size;
    bnn->input_.size = input_length;
    bnn->output_.size = output_length;
    bnn->random_.size = random_size_words;
    bnn->random_.size_in_power_of_two = random_array_length_in_power_of_two;
    bnn->storage_.size = quantity_of_neurons;
    bnn->storage_.size_in_power_of_two = quantity_of_neurons_in_power_of_two;
    bnn->motor_binaries_.size = quantity_of_motor_binaries;
    bnn->motor_binaries_.size_per_motor = quantity_of_motor_binaries_per_motor;
    bnn->threads_.size = threads_size;
    bnn->threads_.size_per_thread = quantity_of_neurons / threads_size;

    bnn->parameters_.quantity_of_neurons_binary = quantity_of_neurons_binary;
    bnn->parameters_.quantity_of_initialized_neurons_binary = 0;
    bnn->parameters_.candidate_for_kill = ~u_word{0};
    bnn->parameters_.start = false;
    bnn->parameters_.stop = false;

    bnn->parameters_.random_config.get_offset = 0;
    bnn->parameters_.random_config.get_offset_in_word = 0;
    bnn->parameters_.random_config.put_offset_start = 0;
    bnn->parameters_.random_config.put_offset_end = random_size_words;
    bnn->parameters_.random_config.put_offset = 0;
    bnn->parameters_.random_config.put_offset_in_word = 0;
#ifdef DEBUG
    bnn->parameters_.random_config.debug_count_get = 0;
    bnn->parameters_.random_config.debug_count_put = 0;
    bnn->parameters_.random_config.debug_sum_put = 0;
#endif

    bnn->memory_.data = memory;
    bnn->input_.data = reinterpret_cast<bool*>(bnn) + sizeof(bnn_bnn);
    bnn->output_.data = bnn->input_.data + input_length;
    bnn->random_.data = reinterpret_cast<u_word*>(bnn->output_.data + output_length);
    bnn->storage_.data = reinterpret_cast<bnn_storage*>(bnn->random_.data + random_size_words);
    bnn->motor_binaries_.data = reinterpret_cast<bnn_motor::binary*>(bnn->storage_.data + quantity_of_neurons);
    bnn->threads_.data = reinterpret_cast<bnn_thread*>(bnn->motor_binaries_.data + quantity_of_motor_binaries);
    void* end = reinterpret_cast<void*>(bnn->threads_.data + threads_size);

    bnn_random_set(
            &bnn->random_,
            &bnn->parameters_.random_config
            );

    auto set_neurons = [](
            bnn_bnn* bnn
            ) -> void
    {
        u_word n = 0;

        auto increment_n = [](
                struct bnn_bnn* bnn,
                u_word* n
                )
        {
            *n += bnn->threads_.size_per_thread;

            if(*n >= bnn->storage_.size)
                *n = *n - bnn->storage_.size + 1;
        };

        for(u_word i = 0; i < bnn->input_.size; ++i)
        {
            bnn->input_.data[i] = bnn_random_pull(&bnn->random_, 1, &bnn->parameters_.random_config);
            bnn_sensor_set(&bnn->storage_.data[n].sensor_, bnn->input_.data, i);
            increment_n(bnn, &n);
        }

        for(u_word i = 0; i < bnn->output_.size; ++i)
        {
            bnn->output_.data[i] = bnn_random_pull(&bnn->random_, 1, &bnn->parameters_.random_config);
            bnn_motor_set(bnn, &bnn->storage_.data[n].motor_, i);
            bnn->storage_.data[n].motor_.binaries_offset = i * bnn->motor_binaries_.size_per_motor;

            bnn_motor::binary b;
            for(u_word j = 0; j < bnn->motor_binaries_.size_per_motor; ++j)
            {
                bnn->motor_binaries_.data[i * bnn->motor_binaries_.size_per_motor + j] = b;
            }

            increment_n(bnn, &n);
        }

        for(u_word i = 0; i < bnn->parameters_.quantity_of_neurons_binary; i++)
        {
            bnn_binary_set(&bnn->storage_.data[n].binary_);
            increment_n(bnn, &n);
        }
    };
    set_neurons(bnn);

    auto set_threads = [](
            bnn_bnn* bnn
            ) -> void
    {
        u_word random_array_length_per_thread = bnn->random_.size / bnn->threads_.size;
        u_word start_neuron = 0;
        u_word length_in_us_in_power_of_two = bnn_math_log2_1(bnn->threads_.size_per_thread);

        bnn_thread thread;
        for(u_word i = 0; i < bnn->threads_.size; i++)
        {
            thread.random_config.put_offset_start = random_array_length_per_thread * i;
            thread.random_config.put_offset = thread.random_config.put_offset_start;
            thread.random_config.put_offset_end = random_array_length_per_thread * (i + 1);
            thread.random_config.get_offset = thread.random_config.put_offset_start;
            thread.length_in_us_in_power_of_two = length_in_us_in_power_of_two;
            thread.start_neuron = start_neuron;
            thread.thread_number = i;
            bnn->threads_.data[i] = thread;
            start_neuron += bnn->threads_.size_per_thread;
        }
    };
    set_threads(bnn);

    *bnn_ = bnn;
};

auto bnn_get_output = [](
        bnn_bnn* bnn,
        u_word offset
        ) -> bool
{
    return bnn->output_.data[offset];
};

auto bnn_set_input = [](
        bnn_bnn* bnn,
        u_word offset,
        bool value
        ) -> void
{
     bnn->input_.data[offset] = value;
};

auto bnn_bnn_function = [](
        bnn_bnn* bnn
        ) -> void
{
    u_word iteration_old = 0, iteration_new = 0, quantity_of_initialized_neurons_binary_temp;

    while(!bnn->parameters_.start)
        ;

    while(!bnn->parameters_.stop)
    {
        if(iteration_old < iteration_new)
        {
            bnn->parameters_.candidate_for_kill = bnn_random_pull(
                        &bnn->random_,
                        bnn->storage_.size_in_power_of_two,
                        &bnn->parameters_.random_config);

            iteration_old = iteration_new;
        }

        iteration_new = 0;
        quantity_of_initialized_neurons_binary_temp = 0;

        for(u_word i = 0; i < bnn->threads_.size; ++i)
        {
            iteration_new += bnn->threads_.data[i].iteration;
            quantity_of_initialized_neurons_binary_temp += bnn->threads_.data[i].quantity_of_initialized_neurons_binary;
        }

        bnn->parameters_.iteration = iteration_new / bnn->threads_.size;
        bnn->parameters_.quantity_of_initialized_neurons_binary = quantity_of_initialized_neurons_binary_temp;
        //usleep(BNN_LITTLE_TIME);
    }

    while(true)
    {
        //        usleep(BNN_LITTLE_TIME);
        for(u_word i = 0; i < bnn->threads_.size; ++i)
            if(bnn->threads_.data[i].in_work)
                continue;

        break;
    }

//    logging("brain stopped");
//    in_work = false;
};

auto bnn_bnn_start = [](
        bnn_bnn* bnn
        ) -> void
{
    bnn->parameters_.stop = false;
    bnn->parameters_.start = true;
};

auto bnn_bnn_stop = [](
        bnn_bnn* bnn
        ) -> void
{
    bnn->parameters_.start = false;
    bnn->parameters_.stop = true;
};

#endif // BNN_BNN_IMPLEMENTATION_H
