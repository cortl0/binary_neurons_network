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

auto bnn_calculate_alignment = [BNN_LAMBDA_REFERENCE](u_word& size) -> void
{
    u_word alignment = size % BNN_BYTES_ALIGNMENT;

    if(alignment > 0)
        alignment = BNN_BYTES_ALIGNMENT - alignment;

    size += alignment;
};

auto bnn_calculate_settings = [BNN_LAMBDA_REFERENCE](
        bnn_bnn* bnn_settings
        ) -> void
{
    bnn_settings->motor_binaries_.size_per_motor = 16;
    bnn_settings->motor_binaries_.size = bnn_settings->motor_binaries_.size_per_motor * bnn_settings->output_.size;
    bnn_settings->storage_.size = bnn_math_two_pow_x(bnn_settings->storage_.size_in_power_of_two);
    bnn_settings->threads_.size = bnn_math_two_pow_x(bnn_settings->threads_.size_in_power_of_two);
    bnn_settings->storage_.size = bnn_settings->storage_.size
            - bnn_settings->input_.size - bnn_settings->output_.size;
    bnn_settings->threads_.neurons_per_thread = bnn_settings->storage_.size / bnn_settings->threads_.size;

    if(bnn_settings->storage_.size <= bnn_settings->input_.size + bnn_settings->output_.size)
    {
        //throw_error("quantity_of_neurons_sensor + quantity_of_neurons_motor >= quantity_of_neurons_end");
        bnn_settings->bnn_error_code = bnn_error_codes::invalid_value;
    }

    auto calculate_random_size_bits = [&]() -> u_word
    {
        bnn_error_codes bnn_error_code;
        const u_word experimental_coefficient = bnn_math_log2_1(&bnn_error_code, QUANTITY_OF_BITS_IN_WORD) + 5;
        bnn_settings->random_.size_in_power_of_two = bnn_settings->storage_.size_in_power_of_two + experimental_coefficient;

        if(bnn_settings->random_.size_in_power_of_two >= QUANTITY_OF_BITS_IN_WORD)
        {
            bnn_settings->random_.size_in_power_of_two = QUANTITY_OF_BITS_IN_WORD - 1;
            //logging("random_array_length_in_power_of_two does not satisfy of experimental_coefficient");
        }

        if(bnn_settings->random_.size_in_power_of_two > 29)
        {
            bnn_settings->random_.size_in_power_of_two = 29;
            //logging("random_array_length_in_power_of_two hardware limitation");
        }

        return (1 << bnn_settings->random_.size_in_power_of_two) / QUANTITY_OF_BITS_IN_WORD;
    };
    bnn_settings->random_.size = calculate_random_size_bits();
    bnn_settings->parameters_.random_config.put_offset_end = bnn_settings->random_.size;

    bnn_settings->memory_.size = 0;
    bnn_settings->memory_.size += sizeof(bnn_bnn);
    bnn_calculate_alignment(bnn_settings->memory_.size);
    bnn_settings->memory_.size += sizeof(bool) * bnn_settings->input_.size;
    bnn_calculate_alignment(bnn_settings->memory_.size);
    bnn_settings->memory_.size += sizeof(bool) * bnn_settings->output_.size;
    bnn_calculate_alignment(bnn_settings->memory_.size);
    bnn_settings->memory_.size += sizeof(u_word) * bnn_settings->random_.size;
    bnn_calculate_alignment(bnn_settings->memory_.size);
    bnn_settings->memory_.size += sizeof(bnn_storage) * bnn_settings->storage_.size;
    bnn_calculate_alignment(bnn_settings->memory_.size);
    bnn_settings->memory_.size += sizeof(bnn_motor::binary) * bnn_settings->motor_binaries_.size;
    bnn_calculate_alignment(bnn_settings->memory_.size);
    bnn_settings->memory_.size += sizeof(bnn_thread) * bnn_settings->threads_.size;
    bnn_calculate_alignment(bnn_settings->memory_.size);
};

auto bnn_calculate_pointers = [BNN_LAMBDA_REFERENCE](
        bnn_bnn* bnn
        ) -> void
{
    u_word size = 0;
    bnn->memory_.data = reinterpret_cast<void*>(bnn);

    size += sizeof(bnn_bnn);
    bnn_calculate_alignment(size);
    bnn->input_.data = reinterpret_cast<bool*>(bnn) + size;

    size += bnn->input_.size;
    bnn_calculate_alignment(size);
    bnn->output_.data = reinterpret_cast<bool*>(bnn) + size;

    size += bnn->output_.size;
    bnn_calculate_alignment(size);
    bnn->random_.data = reinterpret_cast<u_word*>(reinterpret_cast<char*>(bnn) + size);

    size += sizeof(u_word) * bnn->random_.size;
    bnn_calculate_alignment(size);
    bnn->storage_.data = reinterpret_cast<bnn_storage*>(reinterpret_cast<char*>(bnn) + size);

    size += sizeof(bnn_storage) * bnn->storage_.size;
    bnn_calculate_alignment(size);
    bnn->motor_binaries_.data = reinterpret_cast<bnn_motor::binary*>(reinterpret_cast<char*>(bnn) + size);

    size += sizeof(bnn_motor::binary) * bnn->motor_binaries_.size;
    bnn_calculate_alignment(size);
    bnn->threads_.data = reinterpret_cast<bnn_thread*>(reinterpret_cast<char*>(bnn) + size);

    size += sizeof(bnn_thread) * bnn->threads_.size;
    bnn_calculate_alignment(size);
    void* end = reinterpret_cast<void*>(reinterpret_cast<char*>(bnn) + size);
};

auto bnn_shift_pointers = [BNN_LAMBDA_REFERENCE](
        bnn_bnn* bnn,
        int offset
        ) -> void
{
    bnn->memory_.data = reinterpret_cast<void*>(reinterpret_cast<char*>(bnn->memory_.data) + offset);
    bnn->input_.data = reinterpret_cast<bool*>(reinterpret_cast<char*>(bnn->input_.data) + offset);
    bnn->output_.data = reinterpret_cast<bool*>(reinterpret_cast<char*>(bnn->output_.data) + offset);
    bnn->random_.data = reinterpret_cast<u_word*>(reinterpret_cast<char*>(bnn->random_.data) + offset);
    bnn->storage_.data = reinterpret_cast<bnn_storage*>(reinterpret_cast<char*>(bnn->storage_.data) + offset);
    bnn->motor_binaries_.data = reinterpret_cast<bnn_motor::binary*>(reinterpret_cast<char*>(bnn->motor_binaries_.data) + offset);
    bnn->threads_.data = reinterpret_cast<bnn_thread*>(reinterpret_cast<char*>(bnn->threads_.data) + offset);
};

auto bnn_fill_threads = [BNN_LAMBDA_REFERENCE](
        bnn_bnn* bnn
        ) -> void
{
    u_word random_array_length_per_thread = bnn->random_.size / bnn->threads_.size;
    u_word start_neuron = 0;
    bnn_error_codes bnn_error_code;
    u_word length_in_us_in_power_of_two = bnn_math_log2_1(&bnn_error_code, bnn->threads_.neurons_per_thread);

    for(u_word i = 0; i < bnn->threads_.size; i++)
    {
        bnn_thread thread;
        thread.random_config.put_offset_start = random_array_length_per_thread * i;
        thread.random_config.put_offset = thread.random_config.put_offset_start;
        thread.random_config.put_offset_end = random_array_length_per_thread * (i + 1);
        thread.random_config.get_offset = thread.random_config.put_offset_start;
        thread.length_in_us_in_power_of_two = length_in_us_in_power_of_two;
        thread.start_neuron = start_neuron;
        thread.thread_number = i;
        bnn->threads_.data[i] = thread;
        start_neuron += bnn->threads_.neurons_per_thread;
    }
};

auto bnn_fill_random_of_thread = [BNN_LAMBDA_REFERENCE](
        bnn_bnn* bnn,
        u_word thread_number
        ) -> void
{
    bnn_m_sequence m_sequence;
    u_word m_sequence_seed = u_word(1) + (u_word(1) << thread_number % bnn->random_.size_in_power_of_two);
    m_sequence.triggers = m_sequence_seed;
    bnn_m_sequence_set(&m_sequence, bnn->random_.size_in_power_of_two);

    bnn_random_set
            (
                &bnn->random_,
                &bnn->threads_.data[thread_number].random_config,
                &m_sequence
            );
};

auto bnn_set_neurons_of_thread = [BNN_LAMBDA_REFERENCE](
        bnn_bnn* bnn,
        u_word thread_number
        ) -> void
{
    u_word counter_neuron{thread_number * bnn->threads_.neurons_per_thread};

    for(u_word counter_sensor = thread_number; counter_sensor < bnn->input_.size; counter_sensor += bnn->threads_.size)
    {
        bnn->input_.data[counter_sensor] = bnn_random_pull(&bnn->random_, 1, &bnn->threads_.data[thread_number].random_config);
        bnn_sensor_set(&bnn->storage_.data[counter_neuron].sensor_, bnn->input_.data, counter_sensor);
        ++counter_neuron;
    }

    for(u_word counter_motor = thread_number; counter_motor < bnn->output_.size; counter_motor += bnn->threads_.size)
    {
        bnn->output_.data[counter_motor] = bnn_random_pull(&bnn->random_, 1, &bnn->threads_.data[thread_number].random_config);
        bnn_motor_set(bnn, &bnn->storage_.data[counter_neuron].motor_, counter_motor);
        bnn->storage_.data[counter_neuron].motor_.binaries_offset = counter_motor * bnn->motor_binaries_.size_per_motor;

        bnn_motor::binary b;
        for(u_word j = 0; j < bnn->motor_binaries_.size_per_motor; ++j)
        {
            bnn->motor_binaries_.data[counter_motor * bnn->motor_binaries_.size_per_motor + j] = b;
        }

        ++counter_neuron;
    }

    for(; counter_neuron < (thread_number + 1) * bnn->threads_.neurons_per_thread; ++counter_neuron)
    {
        bnn_binary_set(&bnn->storage_.data[counter_neuron].binary_);
    }
};
#if(0) // old
auto bnn_set_neurons_of_thread = [BNN_LAMBDA_REFERENCE](
        bnn_bnn* bnn,
        u_word thread_number
        ) -> void
{
//    u_word counter_sensor = bnn->input_.size / bnn->threads_.size;
//    u_word counter_motor = bnn->output_.size / bnn->threads_.size;
//    u_word counter_neuron{0};

    u_word n = 0;

    auto increment_n = [BNN_LAMBDA_REFERENCE](
            struct bnn_bnn* bnn,
            u_word* n
            )
    {
        *n += bnn->threads_.neurons_per_thread;

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

    for(u_word i = 0; i < bnn->storage_.size; i++)
    {
        bnn_binary_set(&bnn->storage_.data[n].binary_);
        increment_n(bnn, &n);
    }
};
#endif
auto bnn_get_output = [BNN_LAMBDA_REFERENCE](
        bnn_bnn* bnn,
        u_word offset
        ) -> bool
{
    return bnn->output_.data[offset];
};

auto bnn_set_input = [BNN_LAMBDA_REFERENCE](
        bnn_bnn* bnn,
        u_word offset,
        bool value
        ) -> void
{
     bnn->input_.data[offset] = value;
};

auto bnn_bnn_function = [BNN_LAMBDA_REFERENCE](
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

auto bnn_bnn_start = [BNN_LAMBDA_REFERENCE](
        bnn_bnn* bnn
        ) -> void
{
    bnn->parameters_.stop = false;
    bnn->parameters_.start = true;
};

auto bnn_bnn_stop = [BNN_LAMBDA_REFERENCE](
        bnn_bnn* bnn
        ) -> void
{
    bnn->parameters_.start = false;
    bnn->parameters_.stop = true;
};

#endif // BNN_BNN_IMPLEMENTATION_H
