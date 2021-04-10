//*************************************************************//
//                                                             //
//   binary neurons network                                    //
//   created by Ilya Shishkin                                  //
//   cortl@8iter.ru                                            //
//   http://8iter.ru/ai.html                                   //
//   https://github.com/cortl0/binary_neurons_network          //
//   licensed by GPL v3.0                                      //
//                                                             //
//*************************************************************//

#include "../brain.h"

namespace bnn
{

brain::union_storage::binary::binary()
{
    neuron_type_ = neuron_type_binary;
    neuron_binary_type_ = neuron_binary_type_free;
}

void brain::union_storage::binary::init(_word j, _word k, std::vector<union_storage> &us)
{
    neuron_binary_type_ = neuron_binary_type_in_work;
    first = j;
    second = k;
    signals_occupied = 0;
    motor_connect = false;
    motor_consensus = 0;
    us[j].neuron_.signals_occupied++;
    us[k].neuron_.signals_occupied++;
    first_mem = us[j].neuron_.out_new;
    second_mem = us[k].neuron_.out_new;
    solve_body(us);
    out_old = out_new;
    level = us[j].neuron_.level > us[k].neuron_.level ? us[j].neuron_.level + 1 : us[k].neuron_.level + 1;
    kill_factor = 0;
}

bool brain::union_storage::binary::create(brain &brn)
{
    _word j = brn.rndm->get(brn.quantity_of_neurons_in_power_of_two);
    _word k = brn.rndm->get(brn.quantity_of_neurons_in_power_of_two);
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
    if (brn.rndm->get_ft(0, brn.us[j].neuron_.signals_occupied + brn.us[k].neuron_.signals_occupied))
        return false;
    init(j, k, brn.us);
    brn.quantity_of_initialized_neurons_binary++;
    return true;
}

void brain::union_storage::binary::kill(brain &brn)
{
    brn.us[first].neuron_.signals_occupied--;
    brn.us[second].neuron_.signals_occupied--;
    neuron_binary_type_ = neuron_binary_type_marked_to_kill;
    if (motor_connect)
    {
        motor_connect = false;
        (brn.us[motor]).motor_.slots_occupied--;
    }
    brn.debug_soft_kill++;
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

void brain::union_storage::binary::solve(brain &brn)
{
    switch (neuron_binary_type_)
    {
    case binary::neuron_binary_type_free:

#define creating_condition 3

#if(creating_condition == 0)
        // Does not work without conditions
#elif(creating_condition == 1)
        // Well and completely in line with theory
        // But required calculations with high bit number
        if (brn.quantity_of_neurons < brn.rndm->get(brn.quantity_of_neurons_in_power_of_two) *
                (brn.quantity_of_neurons_binary - brn.quantity_of_initialized_neurons_binary))
            // ?? brn.quantity_of_neurons <-> brn.quantity_of_neurons_binary ??
#elif(creating_condition == 2)
        // Well and completely in line with theory
        // But slowly (required operation division), inaccurate due to rounding with integers
        if (brn.quantity_of_neurons / (brn.quantity_of_neurons_binary - brn.quantity_of_initialized_neurons_binary)
                < brn.rndm->get(brn.quantity_of_neurons_in_power_of_two))
            // ?? brn.quantity_of_neurons <-> brn.quantity_of_neurons_binary ??
#elif(creating_condition == 3)
        // Well and quickly
        if (brn.quantity_of_initialized_neurons_binary < brn.rndm->get(brn.quantity_of_neurons_in_power_of_two))
#elif(creating_condition == 4)
        // ??
        if (brn.quantity_of_neurons_binary > brn.rndm->get(brn.quantity_of_neurons_in_power_of_two) *
                brn.quantity_of_initialized_neurons_binary)
#elif(creating_condition == 5)
        // ??
        if (-brn.quantity_of_neurons_binary > (brn.rndm->get(brn.quantity_of_neurons_in_power_of_two) - brn.quantity_of_neurons_binary) *
                (brn.quantity_of_neurons_binary - brn.quantity_of_initialized_neurons_binary))
#elif(creating_condition == 6)
        // ??
        if (brn.quantity_of_neurons_binary < brn.rndm->get_ft(0, brn.quantity_of_neurons_binary) *
                (brn.quantity_of_neurons_binary - brn.quantity_of_initialized_neurons_binary))
#endif
            create(brn);

        break;
    case neuron_binary_type_in_work:
    {
        bool b = false;
        if (brn.us[first].neuron_.get_type() == neuron_type_binary)
            if (brn.us[first].binary_.get_type_binary() == neuron_binary_type_marked_to_kill)
                b = true;
        if (brn.us[second].neuron_.get_type() == neuron_type_binary)
            if (brn.us[second].binary_.get_type_binary() == neuron_binary_type_marked_to_kill)
                b = true;
        if (b)
            kill(brn);
        else
        {
            brn.debug_quantity_of_solve_binary++;
            out_old = out_new;
            solve_body(brn.us);
            if (out_new != out_old)
                brn.rndm->put(out_new);
            if (motor_connect)
            {
                brn.us[motor].motor_.accumulator += (out_new * 2 - 1) * simple_math::sign0(motor_consensus);
                if ((out_new ^ out_old) & (brn.us[motor].neuron_.out_new ^ brn.us[motor].neuron_.out_old))
                {
                    motor_consensus -= ((out_new ^ brn.us[motor].neuron_.out_new) * 2 - 1);
                }
            }

#define killing_condition 3

#if(killing_condition == 0)
            // Does not work without conditions
#elif(killing_condition == 1)
            // Well and completely in line with theory
            // But required calculations with high bit number
            if (brn.quantity_of_neurons > brn.rndm->get(brn.quantity_of_neurons_in_power_of_two) *
                    (brn.quantity_of_neurons_binary - brn.quantity_of_initialized_neurons_binary))
                // ?? brn.quantity_of_neurons <-> brn.quantity_of_neurons_binary ??
#elif(killing_condition == 2)
            // Well and completely in line with theory
            // But slowly (required operation division), inaccurate due to rounding with integers
            if (brn.quantity_of_neurons / (brn.quantity_of_neurons_binary - brn.quantity_of_initialized_neurons_binary)
                    > brn.rndm->get(brn.quantity_of_neurons_in_power_of_two))
                // ?? brn.quantity_of_neurons <-> brn.quantity_of_neurons_binary ??
#elif(killing_condition == 3)
            // Well and quickly
            if (!brn.rndm->get(brn.quantity_of_neurons_in_power_of_two))
#elif(killing_condition == 4)
            // ??
            if (brn.quantity_of_neurons_binary > brn.rndm->get(brn.quantity_of_neurons_in_power_of_two) *
                    (brn.quantity_of_neurons_binary - brn.quantity_of_initialized_neurons_binary))
#elif(killing_condition == 5)
            // ??
            if (brn.quantity_of_neurons_binary > brn.rndm->get_ft(0, brn.quantity_of_neurons_binary) *
                    (brn.quantity_of_neurons_binary - brn.quantity_of_initialized_neurons_binary))
#endif
                if(brn.quantity_of_neurons_in_power_of_two == brn.quantity_of_neurons_in_power_of_two_max)
                    kill_factor++;

            if((255 - kill_factor) * brn.quantity_of_neurons_binary < brn.quantity_of_initialized_neurons_binary * 255)
                kill(brn);
        }
        break;
    }
    case neuron_binary_type_marked_to_kill:
    {
        if (signals_occupied == 0)
        {
            neuron_binary_type_ = neuron_binary_type_free;
            brn.quantity_of_initialized_neurons_binary--;
            brn.debug_soft_kill--;
        }
        break;
    }
    }
}

} // namespace bnn
