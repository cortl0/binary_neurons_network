/*
 *   binary neurons network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#include "brain_tools_web.h"

namespace fs = std::filesystem;

namespace bnn
{

brain_tools_web::~brain_tools_web()
{

}

brain_tools_web::brain_tools_web(const bnn_settings& bs)
    : bnn_tools(bs)
{

}

QString brain_tools_web::brain_get_state()
{
    //debug_out();

    QString qString = "8iter=" + QString::number(get_iteration());
    qString += "\t bits=" + QString::number(bnn->storage_.size_in_power_of_two);
    qString += "\t n_init=" + QString::number(bnn->parameters_.quantity_of_initialized_neurons_binary);
    qString += "\nquantity_of_neuron_binary=" + QString::number(
                bnn->storage_.size - bnn->input_.size - bnn->output_.size) + "\t";
    qString += "quantity_of_neuron_sensor=" + QString::number(bnn->input_.size) + "\t";
    for(uint i = 0; i < 8*16/*quantity_of_neuron_sensor*/; i+=16)
        if(bnn->input_.data[i]) qString += "1"; else qString += "0";
    qString += "\nquantity_of_neuron_motor=" + QString::number(bnn->output_.size) + "\t";
    for(uint i = 0; i < bnn->output_.size; i++)
        if(bnn->output_.data[i]) qString += "1"; else qString += "0";
    /*
    qString += "\nsignals\t";
    for (uint i = 0; i < brain_.quantity_of_neurons_motor; i++)
        qString += QString::number(brain_.storage_[i + brain_.quantity_of_neurons_sensor].motor_.signals_occupied) + "\t";
    qString += "\nslots\t";
    for (uint i = 0; i < brain_.quantity_of_neurons_motor; i++)
        qString += QString::number(brain_.storage_[i + brain_.quantity_of_neurons_sensor].motor_.slots_occupied) + "\t";
    */
    qString += "\naccum\t";
    for(uint i = 0; i < bnn->storage_.size; ++i)
        if(bnn->storage_.data[i].neuron_.type_ == bnn_neuron::type::motor)
            qString += QString::number(bnn->storage_.data[i].motor_.accumulator) + "\t";
    /*
    qString += "\ncountPut=" + QString::number(brain_.rndm->debug_count_put);
    qString += "\tcountGet=" + QString::number(brain_.rndm->debug_count_get);
    */
    return qString;
}

QString brain_tools_web::brain_get_representation()
{
    QString qString;
    u_word s = bnn->input_.size + bnn->output_.size;
    u_word e = bnn->storage_.size;
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
                        consensus += bnn_math_abs(brain_.storage_[i].binary_.motor_consensus);
                        ++count;
                    }
    }
    */
    qString += "consensus=" + QString::number(consensus);
    qString += "\ncount=" + QString::number(count);
    qString += "\nc/c=" + QString::number((static_cast<double>(consensus) / static_cast<double>(count)));
    return qString;
}

std::map<u_word, u_word> brain_tools_web::graphical_representation()
{
    std::vector<int> v;
    std::map<u_word, u_word> m;
    std::map<u_word, u_word>::iterator it;

    for(u_word i = 0; i < bnn->storage_.size; i++)
        if(bnn->storage_.data[i].neuron_.type_ == bnn_neuron::type::binary && bnn->storage_.data[i].binary_.in_work)
        {
            it = m.find(bnn->storage_.data[i].neuron_.level);

            if (it == m.end())
                m.insert(std::make_pair(bnn->storage_.data[i].neuron_.level, 1));
            else
                ++it->second;
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

        bnn_tools::load(in);
    }
}

void brain_tools_web::resize(u_word brainBits_)
{
#if 0
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
#endif
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

        bnn_tools::save(out);
    }
}

void brain_tools_web::stop()
{
    cpu::stop();
}

} // namespace bnn
