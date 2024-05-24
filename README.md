![](img.png)
# Binary Neurons Network (BNN)
This is an attempt to create AI but not AI in the modern sense of the word.  
It is AI in the original meaning coinciding with the meanings of the following synonyms:  
- Artificial general intelligence (AGI);  
- Strong artificial intelligence (Strong AI);  
- Human-level artificial intelligence (HLAI);  
- True artificial intelligence (True AI).

## Looking for a sponsor

## Usage
```
#include <unistd.h>
#include <iostream>

#include "common/architecture.h"

int main()
{
    constexpr bnn_settings bs
    {
        .quantity_of_neurons_in_power_of_two = 12, // 2^12=4096
        .input_length = 31,
        .output_length = 8,
        .motor_binaries_per_motor = 8,
        .random_size_in_power_of_two = 22,
        .quantity_of_threads_in_power_of_two = 1, // 2^1=2
    };

    bnn::architecture bnn(bs);
    bnn.start();
    while(!bnn.is_active());
    bool stop{false};
    std::thread([&stop](){ sleep(1); stop = true; }).detach();

    while(!stop)
    {
        static char input[bs.input_length + 1]{};
        static char output[bs.output_length + 1]{};
        static bool value;

        for (u_word i = 0; i < bs.input_length; i++)
        {
            value = rand() % 2;

            // Put data in BNN
            bnn.set_input(i, value);

            input[i] = value + 48;
        }

        for (u_word i = 0; i < bs.output_length; i++)
        {
            // Get data from BNN
            value = bnn.get_output(i);

            output[i] = value + 48;
        }

        std::cout << "input=" << input << " output=" << output << std::endl;
        usleep(100000);
    }

    bnn.stop();

    return 0;
}
```

## CMake build

## Example projects for BNN
- [./src/examples/minimal/](../master/src/examples/minimal/) - Contains minimal project  
- [./src/examples/web/](../master/src/examples/web/) - The input to the brain is a picture from the browser, and the output is the movement of the gaze of the brain. QT Creator is required  
- https://github.com/cortl0/device - The practical implementation of a physical device with the binary neurons  
- https://github.com/cortl0/device_3d - Testing the BNN in 3d world

## Author
Ilya Shishkin  
e-mail: cortl@8iter.ru

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
