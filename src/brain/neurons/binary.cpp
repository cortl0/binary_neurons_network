/*
 *   binary neurons network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#include "binary.h"
#include "../brain.h"
#include "../thread.h"
#include "../storage.h"

namespace bnn
{

binary::binary()
{
    type_ = neuron::type::binary;
}

void binary::init(brain &brn, _word thread_number, _word j, _word k, std::vector<storage> &us)
{
    in_work = true;
    first = j;
    second = k;
    first_mem = us[j].neuron_.out_new;
    second_mem = us[k].neuron_.out_new;
    solve_body(us);
    out_old = out_new;
    level = us[j].neuron_.level > us[k].neuron_.level ? us[j].neuron_.level + 1 : us[k].neuron_.level + 1;
    first_life_number = us[j].neuron_.life_number;
    second_life_number = us[k].neuron_.life_number;
    brn.threads[thread_number].quantity_of_initialized_neurons_binary++;
#ifdef DEBUG
    brn.threads[thread_number].debug_created++;
#endif
}

void binary::create(brain &brn, _word thread_number)
{
    _word j, k;

    j = brn.random_->get(brn.quantity_of_neurons_in_power_of_two, brn.threads[thread_number].random_config);
    k = brn.random_->get(brn.quantity_of_neurons_in_power_of_two, brn.threads[thread_number].random_config);

    if (j == k)
        return;

    if (&(this->char_reserve_neuron) == &(brn.storage_[j].neuron_.char_reserve_neuron))
        return;

    if (&(this->char_reserve_neuron) == &(brn.storage_[k].neuron_.char_reserve_neuron))
        return;

    if (!((brn.storage_[j].neuron_.get_type() == neuron::type::binary ? brn.storage_[j].binary_.in_work : false) ||
          (brn.storage_[j].neuron_.get_type() == neuron::type::motor) ||
          (brn.storage_[j].neuron_.get_type() == neuron::type::sensor)))
        return;

    if (!((brn.storage_[k].neuron_.get_type() == neuron::type::binary ? brn.storage_[k].binary_.in_work : false) ||
          (brn.storage_[k].neuron_.get_type() == neuron::type::motor) ||
          (brn.storage_[k].neuron_.get_type() == neuron::type::sensor)))
        return;

    if ((brn.storage_[j].neuron_.out_new == brn.storage_[j].neuron_.out_old) || (brn.storage_[k].neuron_.out_new == brn.storage_[k].neuron_.out_old))
        return;

    init(brn, thread_number, j, k, brn.storage_);
}

void binary::kill(brain &brn, _word thread_number)
{
    life_number++;
    in_work = false;
    brn.threads[thread_number].quantity_of_initialized_neurons_binary--;
#ifdef DEBUG
    brn.threads[thread_number].debug_killed++;
#endif
}

void binary::solve_body(std::vector<storage> &us)
{
    static bool solve_tab[2][2][2][2] = {{{{1, 0}, {0, 0}},
                                          {{1, 0}, {1, 1}}},
                                         {{{0, 0}, {1, 0}},
                                          {{1, 1}, {1, 0}}}};

    out_new = solve_tab[first_mem][second_mem]
            [us[first].neuron_.out_new][us[second].neuron_.out_new];
}

void binary::solve(brain &brn, _word thread_number)
{
    if(in_work)
    {
        bool b = false;

        if (brn.storage_[first].neuron_.get_type() == type::binary)
            if (brn.storage_[first].binary_.life_number != first_life_number)
                b = true;

        if (brn.storage_[second].neuron_.get_type() == type::binary)
            if (brn.storage_[second].binary_.life_number != second_life_number)
                b = true;

        if (b)
            kill(brn, thread_number);
        else
        {
            out_old = out_new;

            solve_body(brn.storage_);

            if (out_new != out_old)
                brn.random_->put(out_new, brn.threads[thread_number].random_config);

            if (&(this->char_reserve_neuron) == &(brn.storage_[brn.candidate_for_kill].neuron_.char_reserve_neuron))
                if(brn.quantity_of_initialized_neurons_binary * 3 > brn.quantity_of_neurons * 2)
                    kill(brn, thread_number);
        }
    }
    else
    {
        if((brn.random_->get_under(brn.quantity_of_neurons_binary - brn.quantity_of_initialized_neurons_binary, brn.threads[thread_number].random_config)) ||
                (brn.quantity_of_initialized_neurons_binary * 3 < brn.quantity_of_neurons * 2))
            create(brn, thread_number);
    }
}

} // namespace bnn
