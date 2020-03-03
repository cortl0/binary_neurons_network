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

#include <iostream>
#include "../../brain/brain.h"
#include "m_sequence.h"

static _word random_array_length_in_power_of_two = 28;
static _word quantity_of_neurons_in_power_of_two = 18;
static const _word input_length = 31;
static const _word output_length = 8;
static char c[input_length + output_length + 32];
void clock_cycle_event();
static brain brn(random_array_length_in_power_of_two,
                 quantity_of_neurons_in_power_of_two,
                 input_length,
                 output_length,
                 clock_cycle_event);
static MSequence mSequence(input_length);

// this method will be performed on every beat of the brain
void communication()
{
    int count = 0;
    c[count++] = 'i';
    c[count++] = 'n';
    c[count++] = '=';
    bool value;
    mSequence.Next();
    for (_word i = 0; i < input_length; i++)
    {
        value = mSequence.GetRegisters() & (1 << i);
        c[count++] = value + 48;
        brn.set_in(i, value); // Put data in the brain
    }
    c[count++] = ' ';
    c[count++] = 'o';
    c[count++] = 'u';
    c[count++] = 't';
    c[count++] = '=';
    for (_word i = 0; i < output_length; i++)
        c[count++] = brn.get_out(i) + 48; // Get data from the brain
    c[count++] = '\0';
    std::cout << c << std::endl;
}

void clock_cycle_event()
{
    communication();
}

int main()
{
    bool detach = false;
    brn.start(detach);
    return 0;
}
