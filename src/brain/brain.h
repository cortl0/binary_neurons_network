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

#ifndef BRAIN_H
#define BRAIN_H

#include <unistd.h>
#include <memory>
#include <mutex>
#include <thread>

#include "pch.h"
#include "simple_math.h"
#include "random_put_get.h"

class brain
{
#ifdef BRAIN_FRIEND_H
    friend struct brain_friend;
#endif
    _word quantity_of_neurons_in_bits;
    _word quantity_of_neurons;
    _word quantity_of_neurons_binary;
    _word quantity_of_neurons_sensor;
    _word quantity_of_neurons_motor;
    _word quantity_of_initialized_neurons_binary = 0;
    _word iteration = 0;
    _word reaction_rate = 0;
    _word debug_soft_kill = 0;
    _word debug_quantity_of_solve_binary = 0;
    bool work = false;
    void (*clock_cycle_event)();
    static void thread_work(brain *brn);
    bool *world_input;
    bool *world_output;
    std::thread thrd;
    std::mutex mtx;
    std::unique_ptr<random_put_get> rndm;
    union union_storage;
    union_storage *us;
    struct neuron
    {
        enum neuron_type{
            neuron_type_neuron = 0,
            neuron_type_sensor = 1,
            neuron_type_binary = 2,
            neuron_type_motor = 3
        };
        neuron_type neuron_type_;
        _word level = 1;
        _word signals_occupied = 0;
        bool buzy = false;
        bool out_new;
        bool out_old;
        char char_reserve_neuron[1];//reserve
        neuron();
        neuron_type get_type(){ return neuron_type_; }
    };
    struct binary : neuron
    {
        enum neuron_binary_type{
            neuron_binary_type_free = 0,
            neuron_binary_type_in_work = 1,
            neuron_binary_type_marked_to_kill = 2
        };
        neuron_binary_type neuron_binary_type_ = neuron_binary_type_free;
        _word first;//input adress
        _word second;//input adress
        _word motor;//motor adress
        int motor_consensus;
        bool first_mem;
        bool second_mem;
        bool motor_connect = false;
        char char_reserve_binary[1];//reserve
        binary();
        neuron_binary_type get_type_binary(){ return neuron_binary_type_; }
        void init(_word j, _word k, union_storage *us);
        bool create(brain &brn);
        void kill(brain &brn);
        void solve_body(union_storage *us);
        void solve(brain &brn);
    };
    struct sensor : neuron
    {
        _word world_input_address;
        char char_reserve_sensor[20];//reserve
        sensor(bool *world_input, _word world_input_address);
        void solve(brain &brn);
    };
    struct motor : neuron
    {
        _word world_output_address;
        _word slots_occupied = 0;
        int accumulator = 0;
        char char_reserve_motor[12];//reserve
        motor(bool *world_output, _word world_output_address_);
        void solve(brain &brn, _word me);
    };
    union union_storage
    {
        neuron neuron_;
        binary binary_;
        sensor sensor_;
        motor motor_;
        _word words[10]; // sizeof(Binary) / sizeof(_word) = 10
        union_storage(){}
        ~union_storage(){}
    };
    void stop();
public:
    volatile bool clock_cycle_completed;
    ~brain();
    brain() = delete;
    brain(_word random_array_length_in_bits,
          _word brain_bits,
          _word input_length,
          _word output_length,
          void (*clock_cycle_event)());
    void start(bool detach = true);
    bool get_out(_word offset);
    void set_in(_word offset, bool value);
};

#endif // BRAIN_H