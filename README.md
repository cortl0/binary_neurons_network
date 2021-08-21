![](img.png)
# Binary Neurons Network (BNN)
This is an attempt to create AI but not AI in the modern sense of the word.  
It is AI in the original meaning coinciding with the meanings of the following synonyms:  
- Artificial general intelligence (AGI);  
- Strong artificial intelligence (Strong AI);  
- Human-level artificial intelligence (HLAI);  
- True artificial intelligence (True AI).

## Looking for a sponsor
Development takes time  
Time is money

## Project files
- ./src/brain/brain.* - Contains the brain class - the basis of the project
- ./src/brain/config.h - Contains configuration definitions
- ./src/brain/neurons/binary.cpp - Source file of binary neuron
- ./src/brain/neurons/motor.cpp - Source file of motor neuron
- ./src/brain/neurons/neuron.cpp - Source file of base neuron
- ./src/brain/neurons/sensor.cpp - Source file of sensor neuron
- ./src/brain/m_sequence.* - Contains M-sequence implementation for random_put_get instance initialization
- ./src/brain/random_put_get.* - Contains easy and fast random numbers generator. First you put, then you get random numbers
- ./src/brain/simple_math.h - Contains easy and fast matematics
- ./src/brain/thread.cpp - Source file of thread class for multithreading
- ./src/brain_friend.* - Friendly class for debug, save, load, stop of brain
- ./src/examples/ - Examples directory. See the [./src/examples/README.md](../master/src/examples/) file for details

## Build
- make
- make clean
- make install
- make uninstall
- QT build

Building and installing the library
```
cd ./src/brain; sudo make install;
```

## Usage

main.cpp
```
#include <iostream>

#include "/usr/local/include/bnn/brain.h"

static bnn::brain brn(24, // random_array_length_in_power_of_two
                      14, // quantity_of_neurons_in_power_of_two
                      31, // input_length
                      8   // output_length
                      );

void cycle()
{
    static long int count = 0;

    std::cout << std::endl << "iteration = " << std::to_string(count++) << std::endl;

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
    brn.start();
    while(1)
    {
        usleep(100000);
        cycle();
    }
    return 0;
}
```

Building the application
```
g++ -std=c++17 -pthread main.cpp /usr/local/lib/brain.so
```

## Example projects for BNN
- https://github.com/cortl0/device  
- https://github.com/cortl0/device_3d

## Author
Ilya Shishkin  
mailto:cortl@8iter.ru

## Achievements
- The first mention of BNN in the literature:  
http://gcrinstitute.org  
http://gcrinstitute.org/papers/055_agi-2020.pdf  
(see page 52)

- BNN was placed in the Arctic Code Vault  
https://archiveprogram.github.com/arctic-vault

## Links
- http://8iter.ru/ai.html
- https://t.me/artificial_intelligence_bnn
- https://t.me/artificial_intelligence_bnn_grp
- https://www.youtube.com/watch?v=z-TKgo2b8mg&t
- http://www.gotai.net/forum/default.aspx?threadid=192413
- http://www.gotai.net/forum/default.aspx?threadid=276702
- https://www.cyberforum.ru/ai/thread1834551.html
- https://agi.place/?ai=9&id=24
- https://agi.place/?ai=9&id=26
- https://agi.place/?ai=9&id=34
- https://github.com/cortl0/binary_neurons_network

## License
This project is licensed under the GPL v3.0 - see the LICENSE file for details
