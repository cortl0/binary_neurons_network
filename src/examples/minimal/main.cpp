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

// pick one
#define first_option_communication
//#define second_option_communication

static _word random_array_length_in_power_of_two = 24;
static _word random_max_value_in_power_of_two = 31;
static _word quantity_of_neurons_in_power_of_two = 16;
static const _word input_length = 64;
static const _word output_length = 8;
static char c[input_length + output_length + 32];

#ifdef first_option_communication
static void clock_cycle_handler(void* owner);
#endif

static bnn::brain brn(random_array_length_in_power_of_two,
                      random_max_value_in_power_of_two,
                      quantity_of_neurons_in_power_of_two,
                      input_length,
                      output_length);

static void communication()
{
    int count = 0;
    c[count++] = 'i';
    c[count++] = 'n';
    c[count++] = '=';
    bool value;
    for (_word i = 0; i < input_length; i++)
    {
// random numbers only demonstrates the workable of the algorithm
// don't expect a wow effect without using real data
        value = rand()%2;
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

#ifdef first_option_communication
void clock_cycle_handler(void*)
{
    // This method will be performed on every cycle of the brain
    communication();
}

int main()
{
    bool detach = false;
    brn.start(&brn, clock_cycle_handler, detach);
    return 0;
}
#endif

#ifdef second_option_communication
int main()
{
    brn.start(nullptr, nullptr);
    while(1)
        // this method will be performed on every cycle of the "while(1) communication();" (see below)
        communication();
}
#endif
