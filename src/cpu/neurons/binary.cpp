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

#include "brain.h"
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

void binary::create(brain& b, const u_word thread_number, const u_word me)
{
    u_word first, second;

    first = b.random_->get(b.quantity_of_neurons_in_power_of_two, b.threads[thread_number].random_config);
    second = b.random_->get(b.quantity_of_neurons_in_power_of_two, b.threads[thread_number].random_config);

    if (first == second)
        return;

    if (me == first)
        return;

    if (me == second)
        return;

    if (!((b.storage_[first]->get_type() == neuron::type::binary ? ((neurons::binary*)(b.storage_[first].get()))->in_work : false) ||
          (b.storage_[first]->get_type() == neuron::type::motor) ||
          (b.storage_[first]->get_type() == neuron::type::sensor)))
        return;

    if (!((b.storage_[second]->get_type() == neuron::type::binary ? ((neurons::binary*)(b.storage_[second].get()))->in_work : false) ||
          (b.storage_[second]->get_type() == neuron::type::motor) ||
          (b.storage_[second]->get_type() == neuron::type::sensor)))
        return;

    if ((b.storage_[first]->output_new == b.storage_[first]->output_old) || (b.storage_[second]->output_new == b.storage_[second]->output_old))
        return;

    init(b, thread_number, first, second, b.storage_);
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

    if(in_work)
    {
        bool ft = false;

        if (b.storage_[first_input_address]->get_type() == type::binary)
            if (b.storage_[first_input_address]->life_counter != first_input_life_counter)
                ft = true;

        if (b.storage_[second_input_address]->get_type() == type::binary)
            if (b.storage_[second_input_address]->life_counter != second_input_life_counter)
                ft = true;

        if (ft)
            kill(b, thread_number);
        else
        {
            output_old = output_new;

            solve_body(b.storage_);

            if (output_new != output_old)
                b.random_->put(output_new, b.threads[thread_number].random_config);

            if (me == b.candidate_for_kill)
                if(b.quantity_of_initialized_neurons_binary * 3 > b.quantity_of_neurons * 2)
                    kill(b, thread_number);
        }
    }
    else
    {
        if((b.random_->get_under(b.quantity_of_neurons_binary - b.quantity_of_initialized_neurons_binary, b.threads[thread_number].random_config)) ||
                (b.quantity_of_initialized_neurons_binary * 3 < b.quantity_of_neurons * 2))
            create(b, thread_number, me);
    }
}

} // namespace bnn::neurons
