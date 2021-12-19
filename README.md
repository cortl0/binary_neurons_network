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
g++ -std=c++17 -pthread main.cpp /usr/local/lib/libbnn.so
```

## Example projects for BNN
- [./src/examples/minimal/](../master/src/examples/minimal/) - Contains minimal project  
- [./src/examples/web/](../master/src/examples/web/) - The input to the brain is a picture from the browser, and the output is the movement of the gaze of the brain  
- https://github.com/cortl0/device - The practical implementation of a physical device with the binary neurons  
- https://github.com/cortl0/device_3d - Testing the BNN in 3d world

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
- https://agi.place/?ai=9&id=48
- https://github.com/cortl0/binary_neurons_network

## License
This project is licensed under the GPL v3.0 - see the LICENSE file for details
