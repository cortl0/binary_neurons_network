/*
 *   Binary Neurons Network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#include "brain.h"

#include <unistd.h>
#include <iostream>
#include <thread>

#include "bnn/bnn_implementation.h"

namespace bnn
{
std::thread t;
//thread::thread()
//{
////    logging("");
//}

//thread::thread(brain* brain_,
//               u_word thread_number,
//               u_word start_neuron,
//               u_word length_in_us_in_power_of_two,
//               random::random::config& random_config)
//    : random_config(random_config),
//      length_in_us_in_power_of_two(length_in_us_in_power_of_two),
//      start_neuron(start_neuron),
//      thread_number(thread_number),
//      brain_(brain_)
//{

//}
brain::brain(
        u_word quantity_of_neurons_in_power_of_two,
        u_word input_length,
        u_word output_length,
        u_word threads_count_in_power_of_two
        )
{
        //return;
    //    quantity_of_neurons_in_power_of_two = 4;
    //    input_length = 2;
    //    output_length = 1;
    //    threads_count_in_power_of_two = 0;
    printf("001\n");
    std::cout << (bool)bnn << std::endl;
    std::cout << "(bool)bnn" << std::endl;

        bnn_bnn_set(
                &bnn,
                quantity_of_neurons_in_power_of_two,
                input_length,
                output_length,
                threads_count_in_power_of_two
                );
        std::cout << (bool)bnn << std::endl;
        std::cout << "(bool)bnn" << std::endl;
    printf("002\n");
}

u_word thread_number = 0;
void brain::start()
{
    bnn->parameters_.start = true;
    bnn->parameters_.stop = false;
printf("003\n");

    auto foo = [&](bnn_bnn* bnn, u_word* thread_number)
    {
        bnn_thread_function(
                bnn,
                *thread_number
                );
    };

    t = std::thread(foo, bnn, &thread_number);
    t.detach();


//    if(in_work)
//        return;

//    std::thread(&thread::function, this).detach();
}

void brain::stop()
{
    bnn->parameters_.stop = true;
    bnn->parameters_.start = false;

    while(bnn->threads_.data[0].in_work);

    free(bnn);
}

void brain::set_input(u_word i, bool value)
{
    //bnn->input_.data[i] = value;
}

bool brain::get_output(u_word i)
{
    return bnn->output_.data[i];
}

//void thread::function()
//{
//    try
//    {
//        in_work = true;
//        u_word reaction_rate = 0;
//        u_word j;
//        u_word quantity_of_neurons = brain_->quantity_of_neurons / brain_->threads.size();
//        logging("thread [" + std::to_string(thread_number) + "] started");

//        while(brain_->treads_to_work)
//        {
//            if(!reaction_rate--)
//            {
//                reaction_rate = quantity_of_neurons;
//                iteration++;

//#ifdef DEBUG
//                u_word debug_average_consensus = 0;
//                u_word debug_count = 0;

//                for(u_word i = brain_->threads[thread_number].start_neuron;
//                    i < brain_->threads[thread_number].start_neuron + brain_->quantity_of_neurons / brain_->threads.size(); i++)
//                    if(brain_->storage_[i]->get_type() == neurons::neuron::type::motor)
//                    {
//                        debug_average_consensus += ((neurons::motor*)(brain_->storage_[i].get()))->debug_average_consensus;
//                        debug_count++;
//                    }

//                if(debug_count > 0)
//                    brain_->threads[thread_number].debug_average_consensus = debug_average_consensus / debug_count;

//                if(brain_->threads[thread_number].debug_max_consensus > 0)
//                    brain_->threads[thread_number].debug_max_consensus--;
//#endif
//            }

//            j = start_neuron + brain_->random_->get(length_in_us_in_power_of_two, random_config);
//            brain_->storage_[j]->solve(*brain_, thread_number, j);
//        }
//    }
//    catch (...)
//    {
//        logging("error in thread [" + std::to_string(thread_number) + "]");
//    }

//    logging("thread [" + std::to_string(thread_number) + "] stopped");
//    in_work = false;
//}

} // namespace bnn

//int main()
//{
//    return 0;
//}
