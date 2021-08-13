/*
 *   binary neurons network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef BRAIN_H
#define BRAIN_H

#include <algorithm>
#include <map>
#include <memory>
#include <thread>
#include <vector>
#include <unistd.h>

#include "config.h"
#include <iostream>
#include "random_put_get.h"
#include "simple_math.h"

namespace bnn
{

struct brain
{
#ifdef BRAIN_FRIEND_H
    friend struct brain_friend;
#endif

    enum state
    {
        state_stopped = 0,
        state_to_start = 1,
        state_started = 2,
        state_to_stop = 3
    };

    struct thread
    {
        _word thread_number;
        _word start_neuron;
        _word length_in_us_in_power_of_two;
        _word quantity_of_initialized_neurons_binary = 0;
        _word random_array_length_in_power_of_two;
        std::thread thread_;
        _word iteration = 0;
        bool in_work = false;
#ifdef DEBUG
        unsigned long long int debug_created = 0;
        unsigned long long int debug_killed = 0;
        _word debug_average_consensus = 0;
        _word debug_max_consensus = 0;
        _word debug_max_consensus_binary_num = 0;
        _word debug_max_consensus_motor_num = 0;
#endif
        std::unique_ptr<random_put_get> rndm;
        thread(brain* brn,
               _word thread_number,
               _word start_neuron,
               _word length_in_us_in_power_of_two,
               _word random_array_length_in_power_of_two,
               m_sequence& m_sequence);
        static void function(brain* brn, _word thread_number, _word start_in_us, _word length_in_us_in_power_of_two);
    };

    union union_storage
    {
        struct neuron
        {
            enum neuron_type
            {
                neuron_type_neuron = 0,
                neuron_type_sensor = 1,
                neuron_type_binary = 2,
                neuron_type_motor = 3
            };
            neuron_type neuron_type_;
            _word level = 1;
            _word life_number = 0;
            bool out_new;
            bool out_old;
            char char_reserve_neuron[2]; // reserve
            neuron();
            neuron_type get_type(){ return neuron_type_; }
            void solve(brain &brn, _word me, _word thread_number);
        };

        struct binary : neuron
        {
            enum neuron_binary_type
            {
                neuron_binary_type_free = 0,
                neuron_binary_type_in_work = 1
            };
            neuron_binary_type neuron_binary_type_ = neuron_binary_type_free;
            _word first;  // input adress
            _word second; // input adress
            _word first_life_number; // input life number
            _word second_life_number; // input life number
            bool first_mem;
            bool second_mem;
            char char_reserve_binary[2]; // reserve
            binary();
            neuron_binary_type get_type_binary(){ return neuron_binary_type_; }
            void init(brain &brn, _word thread_number, _word j, _word k, std::vector<union_storage> &us);
            bool create(brain &brn, _word thread_number);
            void kill(brain &brn, _word thread_number);
            void solve_body(std::vector<union_storage> &us);
            void solve(brain &brn, _word thread_number);
        };

        struct motor : neuron
        {
            struct binary_neuron
            {
                _word adress; // binary neuron adress
                _word life_number;
                int consensus = 0;
                binary_neuron(_word adress, _word life_number);
            };

            _word world_output_address;
            int accumulator = 0;
            std::map<_word, binary_neuron>* binary_neurons;

#ifdef DEBUG
            _word debug_average_consensus = 0;
            char char_reserve_motor[8]; // reserve
#elif
            char char_reserve_motor[12]; // reserve
#endif

            motor(std::vector<bool>& world_output, _word world_output_address_);
            void solve(brain &brn, const _word &me, const _word &thread_number);
        };

        struct sensor : neuron
        {
            _word world_input_address;
            char char_reserve_sensor[28]; // reserve
            sensor(std::vector<bool>& world_input, _word world_input_address);
            void solve(brain &brn);
        };

        neuron neuron_;
        binary binary_;
        sensor sensor_;
        motor motor_;
        _word words[sizeof(binary) / sizeof(_word)];
        union_storage(){}
    };

    std::vector<union_storage> us;

    _word quantity_of_neurons_in_power_of_two;
    _word quantity_of_neurons;
    _word quantity_of_neurons_binary;
    _word quantity_of_neurons_sensor;
    _word quantity_of_neurons_motor;
    _word quantity_of_initialized_neurons_binary = 0;
    _word random_array_length_in_power_of_two;
    _word iteration = 0;
    _word candidate_for_kill = 0;
    _word threads_count = 4;
    
    state state_ = state_stopped;
    std::vector<bool> world_input;
    std::vector<bool> world_output;
    std::vector<thread> threads;
    std::thread main_thread;

    static void main_function(brain* brn);
    void primary_filling();
    void stop();
public:
    ~brain();
    brain() = delete;
    brain(_word random_array_length_in_power_of_two,
          _word quantity_of_neurons_in_power_of_two,
          _word input_length,
          _word output_length);
    void start();
    bool get_out(_word offset);
    _word get_output_length();
    _word get_input_length();
    void set_in(_word offset, bool value);
    _word get_iteration();
};

} // namespace bnn

#endif // BRAIN_H
