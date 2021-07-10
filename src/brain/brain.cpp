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

#include "brain.h"

namespace bnn
{

brain::~brain()
{
    stop();
}

brain::brain(_word random_array_length_in_power_of_two,
             _word random_max_value_to_fill_in_power_of_two,
             _word quantity_of_neurons_in_power_of_two,
             _word input_length,
             _word output_length)
    : quantity_of_neurons_in_power_of_two(quantity_of_neurons_in_power_of_two),
      quantity_of_neurons_sensor(input_length),
      quantity_of_neurons_motor(output_length)
{
    rndm.reset(new random_put_get(random_array_length_in_power_of_two, random_max_value_to_fill_in_power_of_two));
    quantity_of_neurons = simple_math::two_pow_x(quantity_of_neurons_in_power_of_two);
    if (quantity_of_neurons <= quantity_of_neurons_sensor + quantity_of_neurons_motor)
        throw ("quantity_of_neurons_sensor + quantity_of_neurons_motor >= quantity_of_neurons_end");
    us.resize(quantity_of_neurons);

    world_input.resize(quantity_of_neurons_sensor);
    for (_word i = 0; i < quantity_of_neurons_sensor; i++)
    {
        world_input[i] = rndm->get(1);
        us[i].sensor_ = union_storage::sensor(world_input, i);
    }

    world_output.resize(quantity_of_neurons_motor);
    for (uint i = 0; i < quantity_of_neurons_motor; i++)
    {
        world_output[i] = rndm->get(1);
        us[i + quantity_of_neurons_sensor].motor_ = union_storage::motor(world_output, i);
    }

    quantity_of_neurons_binary = quantity_of_neurons - quantity_of_neurons_sensor - quantity_of_neurons_motor;
    for (uint i = quantity_of_neurons_sensor + quantity_of_neurons_motor; i < quantity_of_neurons; i++)
        us[i].binary_ = union_storage::binary();

    quantity_of_neurons_in_power_of_two_max = this->quantity_of_neurons_in_power_of_two;
    this->quantity_of_neurons_in_power_of_two = 0;
    quantity_of_neurons = quantity_of_neurons_sensor + quantity_of_neurons_motor;
    quantity_of_neurons_binary = 0;
    update_quantity();
}

void brain::thread_work(brain* brn)
{
    if(brn->state_ != state::state_to_start)
        throw "brn->state_ != state::state_to_start";

    brn->state_ = state::state_started;

    _word j;

    while(brn->state_ != state::state_to_stop)
    {
        if(!brn->reaction_rate--)
        {
            brn->reaction_rate = simple_math::two_pow_x(brn->quantity_of_neurons_in_power_of_two_max);

            brn->candidate_for_kill = brn->rndm->get(brn->quantity_of_neurons_in_power_of_two_max);
            brn->candidate_except_creation = brn->rndm->get(brn->quantity_of_neurons_in_power_of_two_max);

            brn->iteration++;

            if(nullptr != brn->owner_clock_cycle_handler)
                brn->owner_clock_cycle_handler(brn->owner);

            brn->update_quantity();
        }

        j = brn->rndm->get(brn->quantity_of_neurons_in_power_of_two);

        brn->us[j].neuron_.solve(*brn, j);
    }

    brn->state_ = state::state_stopped;
}

void brain::start(void* owner,
                  void (*owner_clock_cycle_handler)(void* owner),
                  bool detach)
{
    if(state_ != state::state_stopped)
        throw "state_ != state::state_stopped";

    reaction_rate = 0;

    this->owner = owner;
    this->owner_clock_cycle_handler = owner_clock_cycle_handler;

    state_ = state::state_to_start;

    thrd = std::thread(thread_work, this);

    if(detach)
        thrd.detach();
    else
        thrd.join();

    while(state_ != state::state_started);
}

void brain::stop()
{
    if(state_ != state::state_started)
        throw "state_ != state::state_started";

    state_ = state::state_to_stop;

    while(state_ != state::state_stopped);
}

bool brain::get_out(_word offset)
{
    return world_output[offset];
}

_word brain::get_output_length()
{
    return quantity_of_neurons_motor;
}

_word brain::get_input_length()
{
    return quantity_of_neurons_sensor;
}

void brain::set_in(_word offset, bool value)
{
    world_input[offset] = value;
}

void brain::update_quantity()
{
    while(((simple_math::two_pow_x(quantity_of_neurons_in_power_of_two)) < (quantity_of_neurons_sensor + quantity_of_neurons_motor + quantity_of_initialized_neurons_binary) * coefficient)
          && quantity_of_neurons_in_power_of_two < quantity_of_neurons_in_power_of_two_max)
    {
        quantity_of_neurons_in_power_of_two++;

        quantity_of_neurons = simple_math::two_pow_x(quantity_of_neurons_in_power_of_two);

        if(quantity_of_neurons < quantity_of_neurons_sensor + quantity_of_neurons_motor)
            quantity_of_neurons = quantity_of_neurons_sensor + quantity_of_neurons_motor;

        quantity_of_neurons_binary = quantity_of_neurons - quantity_of_neurons_sensor - quantity_of_neurons_motor;
    }
}

} // namespace bnn
