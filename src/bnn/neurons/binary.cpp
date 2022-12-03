/*
 *   binary neurons network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#include "binary.h"

#include <algorithm>

#include "bnn.h"
#include "thread.h"
#include "storage.hpp"

namespace bnn::neurons
{

binary::binary()
{
    type_ = neuron::type::binary;
}

void binary::construct(binary* me)
{
    me->type_ = neuron::type::binary;
}

void binary::init(
        brain& b,
        const u_word thread_number,
        const u_word first,
        const u_word second,
        const std::vector<std::shared_ptr<neuron>>& s
        )
{
    first_input_address = first;
    second_input_address = second;
    first_input_memory = s[first]->output_new;
    second_input_memory = s[second]->output_new;
    solve_body(s);
    output_old = output_new;
    level = s[first]->level > s[second]->level ? s[first]->level + 1 : s[second]->level + 1;
    first_input_life_counter = s[first]->life_counter;
    second_input_life_counter = s[second]->life_counter;
    b.threads[thread_number].quantity_of_initialized_neurons_binary++;
#ifdef DEBUG
    b.threads[thread_number].debug_created++;
#endif
    in_work = true;
}

bool binary::create(brain& b, const u_word thread_number, const u_word me)
{
    u_word first, second;

    first = b.random_->get(b.quantity_of_neurons_in_power_of_two, b.threads[thread_number].random_config);
    second = b.random_->get(b.quantity_of_neurons_in_power_of_two, b.threads[thread_number].random_config);

    if (first == second)
        return false;

    if (me == first)
        return false;

    if (me == second)
        return false;

    if (!((b.storage_[first]->get_type() == neuron::type::binary ? ((neurons::binary*)(b.storage_[first].get()))->in_work : false) ||
          (b.storage_[first]->get_type() == neuron::type::motor) ||
          (b.storage_[first]->get_type() == neuron::type::sensor)))
        return false;

    if (!((b.storage_[second]->get_type() == neuron::type::binary ? ((neurons::binary*)(b.storage_[second].get()))->in_work : false) ||
          (b.storage_[second]->get_type() == neuron::type::motor) ||
          (b.storage_[second]->get_type() == neuron::type::sensor)))
        return false;

    if ((b.storage_[first]->output_new == b.storage_[first]->output_old) || (b.storage_[second]->output_new == b.storage_[second]->output_old))
        return false;

    init(b, thread_number, first, second, b.storage_);

    return true;
}

void binary::kill(brain& b, const u_word thread_number)
{
    in_work = false;
    life_counter++;
    b.threads[thread_number].quantity_of_initialized_neurons_binary--;
#ifdef DEBUG
    b.threads[thread_number].debug_killed++;
#endif
}

void binary::solve_body(const std::vector<std::shared_ptr<neuron>>& s)
{
    static constexpr bool solve_tab[2][2][2][2] = {{{{1, 0}, {0, 0}},
                                          {{1, 0}, {1, 1}}},
                                         {{{0, 0}, {1, 0}},
                                          {{1, 1}, {1, 0}}}};

    output_new = solve_tab[first_input_memory][second_input_memory]
            [s[first_input_address]->output_new][s[second_input_address]->output_new];
}

void binary::solve(brain& b, const u_word thread_number, const u_word me)
{
    neuron::solve(b, thread_number, -1);

    auto update_output_new_for_random_filling = [&]() -> void
    {
        output_old = output_new;
        output_new = b.random_->get(1, b.threads[thread_number].random_config);
    };

    if(in_work)
    {
        bool need_to_kill = false;

        if (b.storage_[first_input_address]->get_type() == type::binary)
            if (b.storage_[first_input_address]->life_counter != first_input_life_counter)
                need_to_kill = true;

        if (b.storage_[second_input_address]->get_type() == type::binary)
            if (b.storage_[second_input_address]->life_counter != second_input_life_counter)
                need_to_kill = true;

        if (need_to_kill)
        {
            kill(b, thread_number);
            update_output_new_for_random_filling();
        }
        else
        {
            if((me == b.candidate_for_kill) &&
                (b.quantity_of_initialized_neurons_binary * 3 > b.quantity_of_neurons * 2))
            {
                kill(b, thread_number);
                update_output_new_for_random_filling();
            }
            else
            {
                output_old = output_new;
                solve_body(b.storage_);
            }
        }
    }
    else
    {
        if((b.random_->get_under(b.quantity_of_neurons_binary - b.quantity_of_initialized_neurons_binary, b.threads[thread_number].random_config)) ||
                (b.quantity_of_initialized_neurons_binary * 3 < b.quantity_of_neurons * 2))
            if(!create(b, thread_number, me))
                update_output_new_for_random_filling();
    }

    neuron::put_random(b, thread_number);
}

} // namespace bnn::neurons
