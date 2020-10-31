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

#include <iostream>
#include "../../brain/brain.h"

static _word random_array_length_in_power_of_two = 24;
static _word random_max_value_to_fill_in_power_of_two = 31;
static _word quantity_of_neurons_in_power_of_two = 14;
static const _word input_length = sizeof (int) * 8 - 1;
static const _word output_length = 8;
static char c[input_length + output_length + 32];
void clock_cycle_event(void* owner);
static bnn::brain brn(random_array_length_in_power_of_two,
                 random_max_value_to_fill_in_power_of_two,
                 quantity_of_neurons_in_power_of_two,
                 input_length,
                 output_length,
                 clock_cycle_event);

// M-sequence only demonstrates the workable of the algorithm
// don't expect a wow effect without using real data
static bnn::m_sequence m_seq(input_length);

// this method will be performed on every beat of the brain
void communication()
{
    int count = 0;
    c[count++] = 'i';
    c[count++] = 'n';
    c[count++] = '=';
    bool value;
    m_seq.next();
    for (_word i = 0; i < input_length; i++)
    {
        value = m_seq.get_registers() & (1 << i);
        c[count++] = value + 48;
        // Put data in the brain
        brn.set_in(i, value);
    }
    c[count++] = ' ';
    c[count++] = 'o';
    c[count++] = 'u';
    c[count++] = 't';
    c[count++] = '=';
    for (_word i = 0; i < output_length; i++)
        // Get data from the brain
        c[count++] = brn.get_out(i) + 48;
    c[count++] = '\0';
    std::cout << c << std::endl;
}

void clock_cycle_event(void*)
{
    communication();
}

int main()
{
    bool detach = false;
    brn.start(nullptr, detach);
    return 0;
}
