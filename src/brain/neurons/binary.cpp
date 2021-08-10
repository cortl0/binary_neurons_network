/*
 *   binary neurons network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#include "../brain.h"

namespace bnn
{

brain::union_storage::binary::binary()
{
    neuron_type_ = neuron_type_binary;
    neuron_binary_type_ = neuron_binary_type_free;
}

void brain::union_storage::binary::init(brain &brn, _word thread_number, _word j, _word k, std::vector<union_storage> &us)
{
    neuron_binary_type_ = neuron_binary_type_in_work;
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

bool brain::union_storage::binary::create(brain &brn, _word thread_number)
{
    _word j = brn.threads[thread_number].rndm->get(brn.quantity_of_neurons_in_power_of_two);
    _word k = brn.threads[thread_number].rndm->get(brn.quantity_of_neurons_in_power_of_two);
    if (j == k)
        return false;
    if (&(this->char_reserve_neuron) == &(brn.us[j].neuron_.char_reserve_neuron))
        return false;
    if (&(this->char_reserve_neuron) == &(brn.us[k].neuron_.char_reserve_neuron))
        return false;
    if (!((brn.us[j].neuron_.get_type() == neuron_type_binary?
           brn.us[j].binary_.get_type_binary() == neuron_binary_type_in_work:false)||
          (brn.us[j].neuron_.get_type() == neuron_type_motor) ||
          (brn.us[j].neuron_.get_type() == neuron_type_sensor)))
        return false;
    if (!((brn.us[k].neuron_.get_type() == neuron_type_binary?
           brn.us[k].binary_.get_type_binary() == neuron_binary_type_in_work:false) ||
          (brn.us[k].neuron_.get_type() == neuron_type_motor) ||
          (brn.us[k].neuron_.get_type() == neuron_type_sensor)))
        return false;
    if ((brn.us[j].neuron_.out_new == brn.us[j].neuron_.out_old) || (brn.us[k].neuron_.out_new == brn.us[k].neuron_.out_old))
        return false;

    init(brn, thread_number, j, k, brn.us);
    return true;
}

void brain::union_storage::binary::kill(brain &brn, _word thread_number)
{
    life_number++;
    neuron_binary_type_ = neuron_binary_type_free;
    brn.threads[thread_number].quantity_of_initialized_neurons_binary--;
#ifdef DEBUG
    brn.threads[thread_number].debug_killed++;
#endif
}

void brain::union_storage::binary::solve_body(std::vector<union_storage> &us)
{
    static bool solve_tab[2][2][2][2] = {{{{1, 0}, {0, 0}},
                                          {{1, 0}, {1, 1}}},
                                         {{{0, 0}, {1, 0}},
                                          {{1, 1}, {1, 0}}}};
    out_new = solve_tab[first_mem][second_mem]
            [us[first].neuron_.out_new][us[second].neuron_.out_new];
}

void brain::union_storage::binary::solve(brain &brn, _word thread_number)
{
    _word candidate_for_kill = brn.candidate_for_kill;

    if (&(this->char_reserve_neuron) == &(brn.us[brn.candidate_for_kill].neuron_.char_reserve_neuron))
        while (true)
        {
            _word i = brn.threads[thread_number].rndm->get(brn.quantity_of_neurons_in_power_of_two);

            if(brn.us[i].neuron_.get_type()==brain::union_storage::neuron::neuron_type_binary)
            {
                brn.candidate_for_kill = i;
                break;
            }
        }

    switch (neuron_binary_type_)
    {
    case binary::neuron_binary_type_free:
    {

        if((brn.threads[thread_number].rndm->get_under(brn.quantity_of_neurons_binary - brn.quantity_of_initialized_neurons_binary)) ||
                (brn.quantity_of_initialized_neurons_binary * 3 < brn.quantity_of_neurons * 2))
            create(brn, thread_number);

        break;
    }
    case neuron_binary_type_in_work:
    {
        bool b = false;

        if (brn.us[first].neuron_.get_type() == neuron_type_binary)
            if (brn.us[first].binary_.life_number != first_life_number)
                b = true;

        if (brn.us[second].neuron_.get_type() == neuron_type_binary)
            if (brn.us[second].binary_.life_number != second_life_number)
                b = true;

        if (b)
            kill(brn, thread_number);
        else
        {
            out_old = out_new;

            solve_body(brn.us);

            if (out_new != out_old)
                brn.threads[thread_number].rndm->put(out_new);

            if (&(this->char_reserve_neuron) == &(brn.us[candidate_for_kill].neuron_.char_reserve_neuron))
                if(brn.quantity_of_initialized_neurons_binary * 3 > brn.quantity_of_neurons * 2)
                    kill(brn, thread_number);
        }

        break;
    }
    }
}

} // namespace bnn
