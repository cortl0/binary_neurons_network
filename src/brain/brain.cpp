/*
 *   binary neurons network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#include "brain.h"
#include "thread.h"
#include "storage.h"

namespace bnn
{

brain::~brain()
{
    stop();

    for (_word i = 0; i < quantity_of_neurons; i++)
        if(storage_[i].neuron_.neuron_type_ == neuron::neuron_type::neuron_type_neuron)
            delete storage_[i].motor_.binary_neurons;
}

brain::brain(_word random_array_length_in_power_of_two,
             _word quantity_of_neurons_in_power_of_two,
             _word input_length,
             _word output_length,
             _word threads_count_in_power_of_two)
    : quantity_of_neurons_in_power_of_two(quantity_of_neurons_in_power_of_two),
      quantity_of_neurons_sensor(input_length),
      quantity_of_neurons_motor(output_length),
      random_array_length_in_power_of_two(random_array_length_in_power_of_two),
      threads_count_in_power_of_two(threads_count_in_power_of_two)
{
    quantity_of_neurons = simple_math::two_pow_x(quantity_of_neurons_in_power_of_two);

    quantity_of_neurons_binary = quantity_of_neurons - quantity_of_neurons_sensor - quantity_of_neurons_motor;

    threads_count = simple_math::two_pow_x(threads_count_in_power_of_two);

    if (quantity_of_neurons <= quantity_of_neurons_sensor + quantity_of_neurons_motor)
        throw ("quantity_of_neurons_sensor + quantity_of_neurons_motor >= quantity_of_neurons_end");
    storage_.resize(quantity_of_neurons);

    _word i = 0;
    _word n = 0;
    bool ft = false;

    world_input.resize(quantity_of_neurons_sensor);

    while(i < quantity_of_neurons_sensor)
    {
        world_input[i] = ft = !ft;
        storage_[i].sensor_ = sensor(world_input, i);

        n += quantity_of_neurons / threads_count;

        if(n >= quantity_of_neurons)
            n = n - quantity_of_neurons + 1;

        i++;
    }

    world_output.resize(quantity_of_neurons_motor);
    i = 0;
    while(i < quantity_of_neurons_motor)
    {
        world_output[i] = ft = !ft;
        storage_[i + quantity_of_neurons_sensor].motor_ = motor(world_output, i);

        n += quantity_of_neurons / threads_count;

        if(n >= quantity_of_neurons)
            n = n - quantity_of_neurons + 1;

        i++;
    }

    i = 0;

    while(i < quantity_of_neurons_binary)
    {
        if(storage_[n].neuron_.get_type() != neuron::neuron_type::neuron_type_sensor &&
                storage_[n].neuron_.get_type() != neuron::neuron_type::neuron_type_motor)
            storage_[n].binary_ = binary();

        n += quantity_of_neurons / threads_count;

        if(n >= quantity_of_neurons)
            n = n - quantity_of_neurons + 1;

        i++;
    }
}

void brain::start()
{
    std::cout << "void brain::start()" << std::endl;

    if(state_ != state::stopped)
        throw "state_ != state::state_stopped";

    state_ = state::start;

    _word quantity_of_neurons_of_one_thread = simple_math::two_pow_x(quantity_of_neurons_in_power_of_two) / threads_count;
    _word length_in_us_in_power_of_two = simple_math::log2_1(quantity_of_neurons_of_one_thread);
    _word start_neuron = 0;

    m_sequence m_sequence(_word_bits - 1);

    random_.reset(new random::random(random_array_length_in_power_of_two, m_sequence));

    for(_word i = 0; i < threads_count; i++)
    {
        std::cout << "thrd.push_back(thread(this, i, start_neuron, length_in_us_in_power_of_two));" << std::endl;

        _word random_array_length_per_thread = simple_math::two_pow_x(quantity_of_neurons_in_power_of_two) / threads_count;

        random::config random_config;
        random_config.put_offset_start = random_array_length_per_thread * i;
        random_config.put_offset_end = random_array_length_per_thread * (i + 1);

        threads.push_back(thread(this, i, start_neuron, length_in_us_in_power_of_two, random_config));
        threads[i].thread_.detach();
        start_neuron += quantity_of_neurons_of_one_thread;
    }

    if(!std::any_of(storage_.begin(), storage_.end(), [](storage& u)
    {
                    if(u.neuron_.get_type()==neuron::neuron_type_binary)
                    if(u.binary_.get_type_binary()==binary::neuron_binary_type_in_work)
                    return true;
                    return false;
}))
        primary_filling();
    else
    {
        for(_word i = 0; i < threads_count; i++)
        {
            threads[i].quantity_of_initialized_neurons_binary = 0;
            for(_word j = 0; j < quantity_of_neurons_of_one_thread; j++)
                if(storage_[j + threads[i].start_neuron].neuron_.get_type()==neuron::neuron_type_binary)
                    if(storage_[j + threads[i].start_neuron].binary_.get_type_binary()==binary::neuron_binary_type_in_work)
                        threads[i].quantity_of_initialized_neurons_binary++;
        }
    }

    main_thread = std::thread(function, this);
    main_thread.detach();

    for(_word i = 0; i < threads_count; i++)
        threads[i].in_work = true;
}

void brain::stop()
{
    if(state_!= state::started)
        throw "state != state::state_started";

    state_ = state::stop;

    do
    {
        sleep(1);
    }
    while(std::any_of(threads.begin(), threads.end(), [](const thread& t){ return t.in_work; }));
    
    threads.clear();
    
    state_= state::stopped;
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

_word brain::get_iteration()
{
    return iteration;
}

void brain::function(brain* brn)
{
    _word iteration, old_iteration, quantity_of_initialized_neurons_binary;

    brn->iteration = 0;

    while(true)
    {
        if(std::any_of(brn->threads.begin(), brn->threads.end(), [](const thread& t){ return !t.in_work; }))
            continue;

        break;
    }

    brn->state_ = state::started;

    while(brn->state_ != state::stop)
    {
        iteration = 0;

        quantity_of_initialized_neurons_binary = 0;

#ifdef DEBUG
        unsigned long long int debug_created = 0;
        unsigned long long int debug_killed = 0;
        _word debug_motors_slots_ocupied = 0;
        _word debug_average_level = 0;
        _word debug_max_level = 0;
        _word debug_max_level_binary_num = 0;
        _word debug_average_consensus = 0;
        _word debug_max_consensus = 0;
        _word debug_max_consensus_binary_num = 0;
        _word debug_max_consensus_motor_num = 0;
        unsigned long long int debug_count_get=0;
        unsigned long long int debug_count_put=0;
        long long int debug_sum_put=0;
#endif

        std::for_each(brn->threads.begin(), brn->threads.end(), [&](const thread& t)
        {
            iteration += t.iteration;
            quantity_of_initialized_neurons_binary += t.quantity_of_initialized_neurons_binary;
#ifdef DEBUG
            debug_created += t.debug_created;
            debug_killed += t.debug_killed;

            debug_average_consensus += t.debug_average_consensus;

            if(debug_max_consensus < t.debug_max_consensus)
            {
                debug_max_consensus = t.debug_max_consensus;
                debug_max_consensus_binary_num = t.debug_max_consensus_binary_num;
                debug_max_consensus_motor_num = t.debug_max_consensus_motor_num;
            }

            //            debug_count_get += t.rndm->debug_count_get;
            //            debug_count_put += t.rndm->debug_count_put;
            //            debug_sum_put += t.rndm->debug_sum_put;
#endif
        });

#ifdef DEBUG
        debug_average_consensus /= brn->threads_count;

        _word debug_count = 0;

        for (_word i = 0; i < brn->storage_.size(); i++)
        {
            if(brn->storage_[i].neuron_.get_type() == neuron::neuron_type_motor)
            {
                debug_motors_slots_ocupied += brn->storage_[i].motor_.binary_neurons->size();
            }

            if(brn->storage_[i].neuron_.get_type() == neuron::neuron_type_binary &&
                    brn->storage_[i].binary_.get_type_binary() == binary::neuron_binary_type_in_work)
            {
                debug_average_level += brn->storage_[i].binary_.level;

                debug_count++;

                if(debug_max_level < brn->storage_[i].binary_.level)
                {
                    debug_max_level = brn->storage_[i].binary_.level;
                    debug_max_level_binary_num = i;
                }
            }

            if(debug_count)
                debug_average_level /= debug_count;
        }
#endif

        brn->quantity_of_initialized_neurons_binary = quantity_of_initialized_neurons_binary;

        old_iteration = brn->iteration;

        brn->iteration = iteration / brn->threads.size();

        if(old_iteration < brn->iteration && !(brn->iteration % 128))
        {
            std::string s("\n");

            s += "iteration " + std::to_string(brn->iteration);
            s += " | initialized " + std::to_string(brn->quantity_of_initialized_neurons_binary);

#ifdef DEBUG
            s += " = created " + std::to_string(debug_created);
            s += " - killed " + std::to_string(debug_killed);
            s += " | motor_slots_ocupied " + std::to_string(debug_motors_slots_ocupied);
            s += "\n";
            s += "level     ";
            s += " | average " + std::to_string(debug_average_level);
            s += " | max " + std::to_string(debug_max_level);
            s += " | max_binary_life " + std::to_string(brn->storage_[debug_max_level_binary_num].neuron_.life_number);
            s += " | max_binary_num " + std::to_string(debug_max_level_binary_num);
            s += " | max_binary_calculation_count " + std::to_string(brn->storage_[debug_max_level_binary_num].neuron_.calculation_count);
            s += "\n";
            s += "consensus ";
            s += " | average " + std::to_string(debug_average_consensus);
            s += " | max " + std::to_string(debug_max_consensus);
            s += " | max_motor_num " + std::to_string(debug_max_consensus_motor_num);
            s += " | max_binary_life " + std::to_string(brn->storage_[debug_max_consensus_binary_num].neuron_.life_number);
            s += " | max_binary_num " + std::to_string(debug_max_consensus_binary_num);
            s += " | max_binary_calculation_count " + std::to_string(brn->storage_[debug_max_consensus_binary_num].neuron_.calculation_count);

            s += "\n";
            s += "random    ";
            s += " | count_get " + std::to_string(debug_count_get);
            s += " | count_put " + std::to_string(debug_count_put);
            s += " | sum_put " + std::to_string(debug_sum_put);
#endif

            std::cout << s << std::endl;
        }
    }
}

void brain::primary_filling()
{
    for (_word i = 0; i < storage_.size(); i++)
        if(storage_[i].neuron_.get_type()==neuron::neuron_type_binary)
        {
            candidate_for_kill = i;
            break;
        }

    //_word count = world_input.size() * world_input.size();
    //_word count = quantity_of_neurons / 16;
    _word count = quantity_of_neurons_binary;
    std::cout << std::to_string(count) << std::endl;
    _word i = 0;
    _word n = 0;

    _word thread_number;

    //while(true)
    for (int m1 = 0; m1 < 2; m1++)
        for (int m2 = 0; m2 < 2; m2++)
            for(_word j = 0; j < quantity_of_neurons - 1; j++)
            {
                if(storage_[j].neuron_.get_type() != neuron::neuron_type::neuron_type_sensor)
                    continue;

                for(_word k = j + 1; k < quantity_of_neurons; k++)
                {
                    if(storage_[k].neuron_.get_type() != neuron::neuron_type::neuron_type_sensor)
                        continue;

                    {
                        while(true)
                        {
                            if(i >= count)
                            {
                                return;
                            }

                            i++;

                            if(storage_[n].neuron_.get_type() == neuron::neuron_type::neuron_type_binary)
                            {
                                storage_[j].sensor_.out_new = m1;
                                storage_[k].sensor_.out_new = m2;

                                thread_number = n / (quantity_of_neurons / threads_count);

                                storage_[n].binary_.init(*this, thread_number, j, k, storage_);

                                n += quantity_of_neurons / threads_count;

                                if(n >= quantity_of_neurons)
                                    n = n - quantity_of_neurons + 1;

                                break;
                            }

                            n += quantity_of_neurons / threads_count;

                            if(n >= quantity_of_neurons)
                                n = n - quantity_of_neurons + 1;
                        }
                    }
                }
            }
    std::cout << std::to_string(i) << std::endl;
}

} // namespace bnn
