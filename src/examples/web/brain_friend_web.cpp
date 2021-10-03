/*
 *   binary neurons network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#include "brain_friend_web.h"
#include "../../brain/storage.h"

namespace bnn
{

brain_friend_web::brain_friend_web(bnn::brain &brain_) : brain_friend(brain_)
{

}

QString brain_friend_web::brain_get_state()
{
    QString qString = "8iter=" + QString::number(brain_.iteration);
    qString += "\t bits=" + QString::number(brain_.quantity_of_neurons_in_power_of_two);
    qString += "\t n_init=" + QString::number(brain_.quantity_of_initialized_neurons_binary);
    qString += "\nquantity_of_neuron_binary=" + QString::number(brain_.quantity_of_neurons_binary) + "\t";
    qString += "quantity_of_neuron_sensor=" + QString::number(brain_.quantity_of_neurons_sensor) + "\t";
    for (uint i = 0; i < 8*16/*quantity_of_neuron_sensor*/; i+=16)
        if (brain_.world_input[i]) qString += "1"; else qString += "0";
    qString += "\nquantity_of_neuron_motor=" + QString::number(brain_.quantity_of_neurons_motor) + "\t";
    for (uint i = 0; i < brain_.quantity_of_neurons_motor; i++)
        if (brain_.world_output[i]) qString += "1"; else qString += "0";
    /*
    qString += "\nsignals\t";
    for (uint i = 0; i < brain_.quantity_of_neurons_motor; i++)
        qString += QString::number(brain_.storage_[i + brain_.quantity_of_neurons_sensor].motor_.signals_occupied) + "\t";
    qString += "\nslots\t";
    for (uint i = 0; i < brain_.quantity_of_neurons_motor; i++)
        qString += QString::number(brain_.storage_[i + brain_.quantity_of_neurons_sensor].motor_.slots_occupied) + "\t";
    */
    qString += "\naccum\t";
    for (uint i = 0; i < brain_.quantity_of_neurons_motor; i++)
        qString += QString::number(brain_.storage_[i + brain_.quantity_of_neurons_sensor].motor_.accumulator) + "\t";
    /*
    qString += "\ncountPut=" + QString::number(brain_.rndm->debug_count_put);
    qString += "\tcountGet=" + QString::number(brain_.rndm->debug_count_get);
    */
    return qString;
}

QString brain_friend_web::brain_get_representation()
{
    QString qString;
    _word s = brain_.quantity_of_neurons_sensor + brain_.quantity_of_neurons_motor;
    _word e = brain_.quantity_of_neurons_binary + brain_.quantity_of_neurons_sensor + brain_.quantity_of_neurons_motor;
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

void brain_friend_web::save()
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

        brain_friend::save(out);
    }
}

void brain_friend_web::load()
{
    QString fileName = QFileDialog::getOpenFileName(nullptr,
                                                    "Open Brain", "",
                                                    "Brain (*.brn);;All Files (*)", new QString("*.brn"));
    if (fileName.isEmpty())
        return;
    else
    {
        std::ifstream in(fs::path(fileName.toStdString()), std::ios::binary);

        brain_friend::load(in);
    }
}

void brain_friend_web::stop()
{
    brain_.stop();
}

void brain_friend_web::resize(_word brainBits_)
{
    brain_.stop();
    if(brainBits_ > brain_.quantity_of_neurons_in_power_of_two)
    {
        _word quantity_of_neuron_end_temp = 1 << (brainBits_);
        std::vector<storage> us_temp = std::vector<storage>(quantity_of_neuron_end_temp);
        for(_word i = 0; i < brain_.quantity_of_neurons; i++)
            for(_word j = 0; j < sizeof(storage) / sizeof(_word); j++)
                us_temp[i].words[j] = brain_.storage_[i].words[j];
        for (_word i = brain_.quantity_of_neurons; i < quantity_of_neuron_end_temp; i++)
            us_temp[i].binary_ = binary();
        std::swap(brain_.storage_, us_temp);
        brain_.quantity_of_neurons_in_power_of_two = brainBits_;
        brain_.quantity_of_neurons = quantity_of_neuron_end_temp;
        brain_.quantity_of_neurons_binary = brain_.quantity_of_neurons - brain_.quantity_of_neurons_sensor - brain_.quantity_of_neurons_motor;
        //brain_.reaction_rate = brain_.quantity_of_neurons;
    }
}

std::map<int, int> brain_friend_web::graphical_representation()
{
    std::vector<int> v;
    std::map<int, int> m;
    std::map<int, int>::iterator it;
    for(_word i = 0; i < brain_.quantity_of_neurons; i++)
        if(brain_.storage_[i].neuron_.get_type() == neuron::neuron_type_binary)
            if(brain_.storage_[i].binary_.get_type_binary() == binary::neuron_binary_type_in_work)
            {
                it = m.find(static_cast<int>(brain_.storage_[i].binary_.level));
                if (it == m.end())
                    m.insert(std::make_pair(static_cast<int>(brain_.storage_[i].binary_.level), 1));
                else
                    it->second++;
            }
    return m;
}

} // namespace bnn
