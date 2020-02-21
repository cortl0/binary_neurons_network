//*************************************************************//
//                                                             //
//   network of binary neurons                                 //
//   created by Ilya Shishkin                                  //
//   cortl@8iter.ru                                            //
//   http://8iter.ru/ai.html                                   //
//   https://github.com/cortl0/network_of_binary_neurons_cpp   //
//   licensed by GPL v3.0                                      //
//                                                             //
//*************************************************************//

#include "brain.h"

brain::~brain()
{
    stop();
}
brain::brain(_word random_array_length_in_bits,
             _word brain_bits,
             _word input_length,
             _word output_length,
             void (*clock_cycle_event_)())
    : quantity_of_neurons_in_bits(brain_bits),
      quantity_of_neurons_sensor(input_length),
      quantity_of_neurons_motor(output_length),
      clock_cycle_event(clock_cycle_event_)
{
    rndm.reset(new random_put_get(random_array_length_in_bits));
    quantity_of_neurons = simple_math::two_pow_x(quantity_of_neurons_in_bits);
    reaction_rate = quantity_of_neurons;
    if (quantity_of_neurons <= quantity_of_neurons_sensor + quantity_of_neurons_motor)
        throw ("quantity_of_neurons_sensor + quantity_of_neurons_motor >= quantity_of_neurons_end");
    us.resize(quantity_of_neurons);
    world_input.resize(quantity_of_neurons_sensor);
    for (_word i = 0; i < quantity_of_neurons_sensor; i++)
    {
        world_input[i] = rndm->get(1);
        us[i].sensor_ = sensor(world_input, i);
    }
    world_output.resize(quantity_of_neurons_motor);
    for (uint i = 0; i < quantity_of_neurons_motor; i++)
    {
        world_output[i] = rndm->get(1);
        us[i + quantity_of_neurons_sensor].motor_ = motor(world_output, i);
    }
    quantity_of_neurons_binary = quantity_of_neurons - quantity_of_neurons_sensor - quantity_of_neurons_motor;
    for (uint i = quantity_of_neurons_sensor + quantity_of_neurons_motor; i < quantity_of_neurons; i++)
        us[i].binary_ = binary();
}
brain::neuron::neuron()
{
    neuron_type_ = neuron_type_neuron;
}
brain::binary::binary()
{
    neuron_type_ = neuron_type_binary;
    neuron_binary_type_ = neuron_binary_type_free;
}
void brain::binary::init(_word j, _word k, std::vector<union_storage> &us)
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
}
bool brain::binary::create(brain &brn)
{
    _word j = brn.rndm->get(brn.quantity_of_neurons_in_bits);
    _word k = brn.rndm->get(brn.quantity_of_neurons_in_bits);
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
void brain::binary::kill(brain &brn)
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
void brain::binary::solve_body(std::vector<union_storage> &us)
{
    static bool solve_tab[2][2][2][2] = {{{{1, 0}, {0, 0}},
                                          {{1, 0}, {1, 1}}},
                                         {{{0, 0}, {1, 0}},
                                          {{1, 1}, {1, 0}}}};
    out_new = solve_tab[first_mem][second_mem]
            [us[first].neuron_.out_new][us[second].neuron_.out_new];
}
void brain::binary::solve(brain &brn)
{
    switch (neuron_binary_type_)
    {
    case brain::binary::neuron_binary_type_free:
        create(brn);
        break;
    case brain::binary::neuron_binary_type_in_work:
    {
        bool b = false;
        if (brn.us[first].neuron_.get_type() == brain::neuron::neuron_type_binary)
            if (brn.us[first].binary_.get_type_binary() == brain::binary::neuron_binary_type_marked_to_kill)
                b = true;
        if (brn.us[second].neuron_.get_type() == brain::neuron::neuron_type_binary)
            if (brn.us[second].binary_.get_type_binary() == brain::binary::neuron_binary_type_marked_to_kill)
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
            if (!brn.rndm->get(brn.quantity_of_neurons_in_bits))
                kill(brn);
        }
        break;
    }
    case brain::binary::neuron_binary_type_marked_to_kill:
    {
        if (signals_occupied == 0)
        {
            neuron_binary_type_ = brain::binary::neuron_binary_type_free;
            brn.quantity_of_initialized_neurons_binary--;
            brn.debug_soft_kill--;
        }
        break;
    }
    }
}
brain::sensor::sensor(std::vector<bool>& world_input, _word world_input_address_)
{
    neuron_type_ = neuron_type_sensor;
    world_input_address = world_input_address_;
    out_new = world_input[world_input_address];
    out_old = out_new;
}
void brain::sensor::solve(brain &brn)
{
    out_old = out_new;
    out_new = brn.world_input[world_input_address];
}
brain::motor::motor(std::vector<bool>& world_output, _word world_output_address_)
{
    neuron_type_ = neuron_type_motor;
    world_output_address = world_output_address_;
    out_new = world_output[world_output_address];
    out_old = out_new;
}
void brain::motor::solve(brain &brn, _word me)
{
    out_old = out_new;
    if (accumulator < 0)
        out_new = false;
    else
        out_new = true;
    brn.world_output[world_output_address] = out_new;
    accumulator >>= 1;
    accumulator += (brn.rndm->get(1) << 1) - 1;
    if (brn.rndm->get_ft(0, slots_occupied))
        return;
    _word i = brn.rndm->get(brn.quantity_of_neurons_in_bits);
    if(brn.us[i].neuron_.get_type()==brain::neuron::neuron_type_binary)
        if(brn.us[i].binary_.get_type_binary()==brain::binary::neuron_binary_type_in_work)
            if(!brn.us[i].binary_.motor_connect)
            {
                slots_occupied++;
                brn.us[i].binary_.motor_connect = true;
                brn.us[i].binary_.motor_consensus = 0;
                brn.us[i].binary_.motor = me;
            }
}
void brain::thread_work(brain *brn)
{
    while(true)
    {
        if(!brn->work)
            return;
        if(!brn->reaction_rate)
        {
            brn->reaction_rate = brn->quantity_of_neurons;
            brn->iteration++;
            brn->debug_quantity_of_solve_binary = 0;
            if(!brn->clock_cycle_completed)
                brn->clock_cycle_completed = true;
            if(nullptr != brn->clock_cycle_event)
                brn->clock_cycle_event();
        }
        brn->reaction_rate--;
        _word j = brn->rndm->get(brn->quantity_of_neurons_in_bits);
        switch (brn->us[j].neuron_.get_type())
        {
        case brain::neuron::neuron_type_binary:
            brn->us[j].binary_.solve(*brn);
            break;
        case brain::neuron::neuron_type_sensor:
            brn->us[j].sensor_.solve(*brn);
            break;
        case brain::neuron::neuron_type_motor:
            brn->us[j].motor_.solve(*brn, j);
            break;
        default:
            break;
        }
    }
}
void brain::start(bool detach)
{
    if(work)
    {
        stop();
    }
    clock_cycle_completed = false;
    thrd = std::thread(thread_work, this);
    work = true;
    if(detach)
        thrd.detach();
    else
        thrd.join();
}
void brain::stop()
{
    mtx.lock();
    work = false;
    mtx.unlock();
    usleep(200);
    clock_cycle_completed = false;
}
bool brain::get_out(_word offset)
{
    return  world_output[offset];
}
void brain::set_in(_word offset, bool value)
{
    world_input[offset] = value;
}
