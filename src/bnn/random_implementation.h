/*
 *   binary neurons network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef BNN_RANDOM_IMPLEMENTATION_H
#define BNN_RANDOM_IMPLEMENTATION_H

#include "random.h"
#include "m_sequence.h"
#include "math.h"

auto bnn_random_push = [BNN_LAMBDA_REFERENCE](
        bnn_random* random,
        const bool i,
        bnn_random_config* config
        ) -> void
{
#ifdef DEBUG
    config->debug_sum_put += i * 2 - 1;
    config->debug_count_put++;
#endif

    random->data[config->put_offset] =
            (random->data[config->put_offset] & (~(1 << config->put_offset_in_word)))
            | (static_cast<u_word>(i) << config->put_offset_in_word);

    config->put_offset_in_word++;

    if(config->put_offset_in_word >= QUANTITY_OF_BITS_IN_WORD)
    {
        config->put_offset_in_word = 0;
        config->put_offset++;

        if(config->put_offset >= config->put_offset_end)
            config->put_offset = 0;
    }
};

auto bnn_random_set = [BNN_LAMBDA_REFERENCE](
        bnn_random* random,
        bnn_random_config* config,
        bnn_m_sequence* m_sequence
        ) -> void
{
    bnn_error_codes bnn_error_code;
    if((random->size_in_power_of_two < bnn_math_log2_1(&bnn_error_code, QUANTITY_OF_BITS_IN_WORD)) ||
            (random->size_in_power_of_two >= QUANTITY_OF_BITS_IN_WORD))
    {
        random->bnn_error_code = bnn_error_codes::invalid_value;
    }

    for(u_word i = config->put_offset_start; i < config->put_offset_end; ++i)
        for(u_word j = 0; j < QUANTITY_OF_BITS_IN_WORD; ++j)
            bnn_random_push(random, bnn_m_sequence_next(m_sequence), config);
};

auto bnn_random_pull = [BNN_LAMBDA_REFERENCE](
        bnn_random* random,
        u_word bits,
        bnn_random_config* config
        ) -> u_word
{
#ifdef DEBUG
    config->debug_count_get += bits;
#endif

    u_word shift, quantity, data, returnValue = 0;

    while(bits)
    {
        quantity = QUANTITY_OF_BITS_IN_WORD - config->get_offset_in_word;
        quantity = quantity < bits ? quantity : bits;
        data = random->data[config->get_offset] >> config->get_offset_in_word;
        shift = QUANTITY_OF_BITS_IN_WORD - quantity;
        data <<= shift;
        data >>= shift;
        returnValue <<= quantity;
        returnValue |= data;
        bits -= quantity;
        config->get_offset_in_word += quantity;

        if(config->get_offset_in_word >= QUANTITY_OF_BITS_IN_WORD)
        {
            config->get_offset_in_word -= QUANTITY_OF_BITS_IN_WORD;
            config->get_offset++;

            if(config->get_offset >= random->size)
                config->get_offset = 0;
        }
    }

    return returnValue;
};

auto bnn_random_pull_under = [BNN_LAMBDA_REFERENCE](
        bnn_random* random,
        u_word to,
        bnn_random_config* config
        ) -> u_word
{
    u_word count = 0;

    while(to >>= 1)
        count++;

    return bnn_random_pull(random, count, config);
};

#endif // BNN_RANDOM_IMPLEMENTATION_H
