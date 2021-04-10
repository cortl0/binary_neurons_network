![](img.png)
# Binary neurons network
This is an attempt to create AI but not AI in the modern sense of the word.  
It is AI in the original meaning coinciding with the meanings of the following synonyms:  
- Artificial general intelligence (AGI);  
- Strong artificial intelligence (Strong AI);  
- Human-level artificial intelligence (HLAI);  
- True artificial intelligence (True AI).

## Project directories

#### ./src/brain/
This folder contains everything you need for a project based on binary neurons network

##### brain.h, brain.cpp
Contains the brain class - the basis of the project

##### config.h
Contains configuration definitions

##### m_sequence.h, m_sequence.cpp  
M-sequence implementation for random_put_get instance initialization

##### random_put_get.h, random_put_get.cpp  
Contains the random_put_get class. Easy and fast random numbers.  
First you put, then you get random numbers.

##### simple_math.h
Contains the simple_math struct. Easy and fast matematics.

#### ./src/brain/neurons/*.cpp
neuron source files: neuron(base), binary neuron, motor neuron, sensor neuron

#### ./src/brain_friend.*
Friendly class for debug, save, load, stop brain

#### ./src/examples/
Examples directory. See the [./src/examples/README.md](../master/src/examples/) file for details

## Example projects
https://github.com/cortl0/device  
https://github.com/cortl0/device_3d

## Build
make  
make clean  
make install  
make uninstall  
QT build

## Usage

main.cpp
```
#include <iostream>

#include "/usr/local/include/bnn/brain.h"

void clock_cycle_handler(void*);

static bnn::brain brn(24, // random_array_length_in_power_of_two,
                      31, // random_max_value_to_fill_in_power_of_two,
                      14, // quantity_of_neurons_in_power_of_two,
                      31, // input_length,
                      8,  // output_length,
                      clock_cycle_handler);

// This method will be performed on every beat of the brain
void clock_cycle_handler(void*)
{
	static long int count = 0;

    std::cout << std::endl << "cycle = " << std::to_string(count++) << std::endl;

    for (_word i = 0; i < brn.get_input_length(); i++)
    {
        // Put your data here
        bool value = true;
        brn.set_in(i, value);
    }

    for (_word i = 0; i < brn.get_output_length(); i++)
    {
        // Get the result from the brain
        bool value = brn.get_out(i);
    }
}

int main()
{
    bool detach = false;
    brn.start(nullptr, detach);
    return 0;
}
```

Building and installing the library
```
cd ./src/brain; sudo make install;
```

Building the application
```
g++ -std=c++17 -pthread main.cpp /usr/local/lib/brain.so
```

## Author
Ilya Shishkin  
mailto:cortl@8iter.ru

## Links
http://8iter.ru/ai.html  
https://t.me/artificial_intelligence_bnn  
https://t.me/artificial_intelligence_bnn_grp  
https://www.youtube.com/watch?v=z-TKgo2b8mg&t  
https://github.com/cortl0/binary_neurons_network

## License
This project is licensed under the GPL v3.0 - see the LICENSE file for details
