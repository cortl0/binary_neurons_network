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

#define first_option_communication
//#define second_option_communication
//#define third_option_communication

static _word random_array_length_in_power_of_two = 28;
static _word quantity_of_neurons_in_power_of_two = 18;
static const _word input_length = 64;
static const _word output_length = 8;
static char c[input_length + output_length + 32];
void clock_cycle_event();
static brain brn(random_array_length_in_power_of_two,
                 quantity_of_neurons_in_power_of_two,
                 input_length,
                 output_length,
                 clock_cycle_event);

// this method will be performed on every beat of the brain
void communication()
{
    int count = 0;
    c[count++] = 'i';
    c[count++] = 'n';
    c[count++] = '=';
    bool value;
    for (_word i = 0; i < input_length; i++)
    {
        value = rand()%2;
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
#ifdef first_option_communication
    communication();
#endif
}

int main()
{
#ifdef first_option_communication
    bool detach = false;
    brn.start(detach);
    return 0;
#endif

#ifdef second_option_communication
    brn.start();
    while(1)
        if (brn.clock_cycle_completed == true)
        {
            communication();
            brn.clock_cycle_completed = false;
        }
#endif

#ifdef third_option_communication
    brn.start();
    while(1)
        communication();
#endif
}
