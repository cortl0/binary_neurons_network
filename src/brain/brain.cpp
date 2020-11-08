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
             _word output_length,
             void (*clock_cycle_handler)(void* owner))
    : quantity_of_neurons_in_power_of_two(quantity_of_neurons_in_power_of_two),
      quantity_of_neurons_sensor(input_length),
      quantity_of_neurons_motor(output_length),
      clock_cycle_handler(clock_cycle_handler)
{
    rndm.reset(new random_put_get(random_array_length_in_power_of_two, random_max_value_to_fill_in_power_of_two));
    quantity_of_neurons = simple_math::two_pow_x(quantity_of_neurons_in_power_of_two);
    reaction_rate = quantity_of_neurons;
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

    //    uint i = quantity_of_neurons_sensor + quantity_of_neurons_motor;
    //    uint j = 0;
    //    while((i < quantity_of_neurons - 4) && (j < quantity_of_neurons_sensor + quantity_of_neurons_motor - 1))
    //    {
    //        for(int n = 0; n < 1; n++)
    //        {
    //            us[j].neuron_.out_new = 0;
    //            us[j + 1].neuron_.out_new = 1;
    //            us[i + n].binary_.init(j, j + 1, us);

    //            us[j].neuron_.out_new = 1;
    //            us[j + 1].neuron_.out_new = 0;
    //            us[i + n + 4].binary_.init(j, j + 1, us);

    //            us[j].neuron_.out_new = 1;
    //            us[j + 1].neuron_.out_new = 1;
    //            us[i + n + 8].binary_.init(j, j + 1, us);

    //            us[j].neuron_.out_new = 0;
    //            us[j + 1].neuron_.out_new = 0;
    //            us[i + n + 12].binary_.init(j, j + 1, us);

    //            quantity_of_initialized_neurons_binary += 4;
    //        }
    //        j++;
    //        i += 4;
    //    }
}

void brain::thread_work(brain* brn)
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
            if(nullptr != brn->clock_cycle_handler)
                brn->clock_cycle_handler(brn->owner);
            brn->update_quantity();
        }
        brn->reaction_rate--;
        _word j = brn->rndm->get(brn->quantity_of_neurons_in_power_of_two);
        brn->us[j].neuron_.solve(*brn, j);
    }
}

void brain::start(void* owner_, bool detach)
{
    if(work)
    {
        stop();
    }
    clock_cycle_completed = false;
    thrd = std::thread(thread_work, this);
    owner = owner_;
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
    while((((_word)1 << quantity_of_neurons_in_power_of_two) < (quantity_of_neurons_sensor + quantity_of_neurons_motor + quantity_of_initialized_neurons_binary) * coefficient)
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
