/*
 *   binary neurons network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#include "brain_tools_web.h"
#include "../../brain/storage.hpp"

namespace bnn
{

brain_tools_web::~brain_tools_web()
{

}

brain_tools_web::brain_tools_web(u_word random_array_length_in_power_of_two,
                                 u_word quantity_of_neurons_in_power_of_two,
                                 u_word input_length,
                                 u_word output_length,
                                 u_word threads_count_in_power_of_two)
    : brain_tools(random_array_length_in_power_of_two,
                  quantity_of_neurons_in_power_of_two,
                  input_length,
                  output_length,
                  threads_count_in_power_of_two)
{

}

QString brain_tools_web::brain_get_state()
{
    //debug_out();

    QString qString = "8iter=" + QString::number(get_iteration());
    qString += "\t bits=" + QString::number(quantity_of_neurons_in_power_of_two);
    qString += "\t n_init=" + QString::number(quantity_of_initialized_neurons_binary);
    qString += "\nquantity_of_neuron_binary=" + QString::number(quantity_of_neurons_binary) + "\t";
    qString += "quantity_of_neuron_sensor=" + QString::number(quantity_of_neurons_sensor) + "\t";
    for (uint i = 0; i < 8*16/*quantity_of_neuron_sensor*/; i+=16)
        if (world_input[i]) qString += "1"; else qString += "0";
    qString += "\nquantity_of_neuron_motor=" + QString::number(quantity_of_neurons_motor) + "\t";
    for (uint i = 0; i < quantity_of_neurons_motor; i++)
        if (world_output[i]) qString += "1"; else qString += "0";
    /*
    qString += "\nsignals\t";
    for (uint i = 0; i < brain_.quantity_of_neurons_motor; i++)
        qString += QString::number(brain_.storage_[i + brain_.quantity_of_neurons_sensor].motor_.signals_occupied) + "\t";
    qString += "\nslots\t";
    for (uint i = 0; i < brain_.quantity_of_neurons_motor; i++)
        qString += QString::number(brain_.storage_[i + brain_.quantity_of_neurons_sensor].motor_.slots_occupied) + "\t";
    */
    qString += "\naccum\t";
    for (uint i = 0; i < quantity_of_neurons_motor; i++)
        qString += QString::number(storage_[i + quantity_of_neurons_sensor].motor_.accumulator) + "\t";
    /*
    qString += "\ncountPut=" + QString::number(brain_.rndm->debug_count_put);
    qString += "\tcountGet=" + QString::number(brain_.rndm->debug_count_get);
    */
    return qString;
}

QString brain_tools_web::brain_get_representation()
{
    QString qString;
    u_word s = quantity_of_neurons_sensor + quantity_of_neurons_motor;
    u_word e = quantity_of_neurons_binary + quantity_of_neurons_sensor + quantity_of_neurons_motor;
    int consensus = 0;
    int count = 0;
    /*
    for (_word i = s; i < e; i++)
    {
        if (brain_.storage_[i].neuron_.get_type() == brain::union_storage::neuron::neuron_type::neuron_type_binary)
            if (brain_.storage_[i].binary_.get_type_binary() == brain::union_storage::binary::neuron_binary_type::neuron_binary_type_in_work)
                if (brain_.storage_[i].binary_.get_type_binary() == brain::union_storage::binary::neuron_binary_type::neuron_binary_type_in_work)
                    if (brain_.storage_[i].binary_.motor_connect)
                    {
                        consensus += simple_math::abs(brain_.storage_[i].binary_.motor_consensus);
                        count++;
                    }
    }
    */
    qString += "consensus=" + QString::number(consensus);
    qString += "\ncount=" + QString::number(count);
    qString += "\nc/c=" + QString::number((static_cast<double>(consensus) / static_cast<double>(count)));
    return qString;
}

std::map<int, int> brain_tools_web::graphical_representation()
{
    std::vector<int> v;
    std::map<int, int> m;
    std::map<int, int>::iterator it;
    for(u_word i = 0; i < quantity_of_neurons; i++)
        if(storage_[i].neuron_.get_type() == neurons::neuron::type::binary && storage_[i].binary_.in_work)
            {
                it = m.find(static_cast<int>(storage_[i].binary_.level));
                if (it == m.end())
                    m.insert(std::make_pair(static_cast<int>(storage_[i].binary_.level), 1));
                else
                    it->second++;
            }
    return m;
}

void brain_tools_web::load()
{
    QString fileName = QFileDialog::getOpenFileName(nullptr,
                                                    "Open Brain", "",
                                                    "Brain (*.brn);;All Files (*)", new QString("*.brn"));
    if (fileName.isEmpty())
        return;
    else
    {
        std::ifstream in(fs::path(fileName.toStdString()), std::ios::binary);

        brain_tools::load(in);
    }
}

void brain_tools_web::resize(u_word brainBits_)
{
    brain::stop();
    if(brainBits_ > quantity_of_neurons_in_power_of_two)
    {
        u_word quantity_of_neuron_end_temp = 1 << (brainBits_);
        std::vector<storage> us_temp = std::vector<storage>(quantity_of_neuron_end_temp);
        for(u_word i = 0; i < quantity_of_neurons; i++)
            for(u_word j = 0; j < sizeof(storage) / sizeof(u_word); j++)
                us_temp[i].words[j] = storage_[i].words[j];
        for (u_word i = quantity_of_neurons; i < quantity_of_neuron_end_temp; i++)
            us_temp[i].binary_ = neurons::binary();
        std::swap(storage_, us_temp);
        quantity_of_neurons_in_power_of_two = brainBits_;
        quantity_of_neurons = quantity_of_neuron_end_temp;
        quantity_of_neurons_binary = quantity_of_neurons - quantity_of_neurons_sensor - quantity_of_neurons_motor;
        //brain_.reaction_rate = brain_.quantity_of_neurons;
    }
}

void brain_tools_web::save()
{
    QString fileName = QFileDialog::getSaveFileName(nullptr,
                                                    "Save Brain", "",
                                                    "Brain (*.brn);;All Files (*)");
    if (fileName.isEmpty())
        return;
    else {
        if (fileName.split('.')[fileName.split('.').length() - 1] != "brn")
            fileName += ".brn";

        std::ofstream out(fs::path(fileName.toStdString()), std::ios::binary);

        brain_tools::save(out);
    }
}

void brain_tools_web::stop()
{
    brain::stop();
}

} // namespace bnn
