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

#include "brain_friend.h"

QString brain_friend::brain_get_state()
{
    QString qString = "8iter=" + QString::number(brain_.iteration);
    qString += "\t bits=" + QString::number(brain_.quantity_of_neurons_in_bits);
    qString += "\t n_init=" + QString::number(brain_.quantity_of_initialized_neurons_binary);
    qString += "\nquantity_of_neuron_binary=" + QString::number(brain_.quantity_of_neurons_binary) + "\t";
    qString += "quantity_of_neuron_sensor=" + QString::number(brain_.quantity_of_neurons_sensor) + "\t";
    for (uint i = 0; i < 8*16/*quantity_of_neuron_sensor*/; i+=16)
        if (brain_.world_input[i]) qString += "1"; else qString += "0";
    qString += "\nquantity_of_neuron_motor=" + QString::number(brain_.quantity_of_neurons_motor) + "\t";
    for (uint i = 0; i < brain_.quantity_of_neurons_motor; i++)
        if (brain_.world_output[i]) qString += "1"; else qString += "0";
    qString += "\nsignals\t";
    for (uint i = 0; i < brain_.quantity_of_neurons_motor; i++)
        qString += QString::number(brain_.us[i + brain_.quantity_of_neurons_sensor].motor_.signals_occupied) + "\t";
    qString += "\nslots\t";
    for (uint i = 0; i < brain_.quantity_of_neurons_motor; i++)
        qString += QString::number(brain_.us[i + brain_.quantity_of_neurons_sensor].motor_.slots_occupied) + "\t";
    qString += "\naccum\t";
    for (uint i = 0; i < brain_.quantity_of_neurons_motor; i++)
        qString += QString::number(brain_.us[i + brain_.quantity_of_neurons_sensor].motor_.accumulator) + "\t";
    qString += "\ncountPut=" + QString::number(brain_.rndm->debug_count_put);
    qString += "\tcountGet=" + QString::number(brain_.rndm->debug_count_get);
    return qString;
}
void brain_friend::save()
{
    QString fileName = QFileDialog::getSaveFileName(nullptr,
                                                    "Save Brain", "",
                                                    "Brain (*.brn);;All Files (*)");
    if (fileName.isEmpty())
        return;
    else {
        if (fileName.split('.')[fileName.split('.').length() - 1] != "brn")
            fileName += ".brn";
        QFile file(fileName);
        if (!file.open(QIODevice::WriteOnly)) {
            QMessageBox::information(nullptr, "Unable to open file",
                                     file.errorString());
            return;
        }
        QDataStream out(&file);
        out.setVersion(QDataStream::Qt_4_5);
        out << version;
        out << brain_.quantity_of_neurons_in_bits;
        out << brain_.quantity_of_neurons;
        out << brain_.quantity_of_neurons_binary;
        out << brain_.quantity_of_neurons_sensor;
        out << brain_.quantity_of_neurons_motor;
        out << brain_.work;
        out << brain_.iteration;
        out << brain_.quantity_of_initialized_neurons_binary;
        out << brain_.debug_soft_kill;
        for(_word i = 0; i < brain_.quantity_of_neurons_sensor; i++)
            out << brain_.world_input[i];
        for(_word i = 0; i < brain_.quantity_of_neurons_motor; i++)
            out << brain_.world_output[i];
        for(_word i = 0; i < brain_.quantity_of_neurons; i++)
            for(_word j = 0; j < sizeof(brain::union_storage) / sizeof(_word); j++)
                out << brain_.us[i].words[j];
        out << brain_.rndm->get_length();
        for(_word i = 0; i < brain_.rndm->get_length(); i++)
            out << brain_.rndm->get_array()[i];
        out << brain_.rndm->debug_count_put;
        out << brain_.rndm->debug_count_get;
    }
}
void brain_friend::load()
{
    QString fileName = QFileDialog::getOpenFileName(nullptr,
                                                    "Open Brain", "",
                                                    "Brain (*.brn);;All Files (*)", new QString("*.brn"));
    if (fileName.isEmpty())
        return;
    else
    {
        QFile file(fileName);
        if (!file.open(QIODevice::ReadOnly)) {
            QMessageBox::information(nullptr, "Unable to open file",
                                     file.errorString());
            return;
        }
        QDataStream in(&file);
        in.setVersion(QDataStream::Qt_4_5);
        QString versionTemp;
        in >> versionTemp;
        if(versionTemp!=version)
        {
            QMessageBox::information(nullptr, "Version mismatch", "Version mismatch");
            return;
        }
        in >> brain_.quantity_of_neurons_in_bits;
        in >> brain_.quantity_of_neurons;
        in >> brain_.quantity_of_neurons_binary;
        in >> brain_.quantity_of_neurons_sensor;
        in >> brain_.quantity_of_neurons_motor;
        in >> brain_.work;
        in >> brain_.iteration;
        in >> brain_.quantity_of_initialized_neurons_binary;
        in >> brain_.debug_soft_kill;
        delete [] brain_.world_input;
        brain_.world_input = new bool[brain_.quantity_of_neurons_sensor];
        for(_word i = 0; i < brain_.quantity_of_neurons_sensor; i++)
            in >> brain_.world_input[i];
        delete [] brain_.world_output;
        brain_.world_output = new bool[brain_.quantity_of_neurons_motor];
        for(_word i = 0; i < brain_.quantity_of_neurons_motor; i++)
            in >> brain_.world_output[i];
        delete [] brain_.us;
        brain_.us = new brain::union_storage[brain_.quantity_of_neurons];
        for(_word i = 0; i < brain_.quantity_of_neurons; i++)
            for(_word j = 0; j < sizeof(brain::union_storage) / sizeof(_word); j++)
                in >> brain_.us[i].words[j];
        _word rndmLength;
        in >> rndmLength;
        if(rndmLength!=brain_.rndm->get_length())
            brain_.rndm.reset(new random_put_get(rndmLength));
        for(_word i = 0; i < brain_.rndm->get_length(); i++)
            in >> brain_.rndm->get_array()[i];
        in >> brain_.rndm->debug_count_put;
        in >> brain_.rndm->debug_count_get;
    }
}
void brain_friend::stop()
{
    brain_.stop();
}
void brain_friend::resize(_word brainBits_)
{
    brain_.mtx.lock();
    brain_.work = false;
    brain_.mtx.unlock();
    usleep(200);
    brain_.mtx.lock();
    if(brainBits_ > brain_.quantity_of_neurons_in_bits)
    {
        _word quantity_of_neuron_end_temp = 1 << (brainBits_);
        brain::union_storage* us_temp = new brain::union_storage[quantity_of_neuron_end_temp];
        for(_word i = 0; i < brain_.quantity_of_neurons; i++)
            for(_word j = 0; j < sizeof(brain::union_storage) / sizeof(_word); j++)
                us_temp[i].words[j] = brain_.us[i].words[j];
        for (_word i = brain_.quantity_of_neurons; i < quantity_of_neuron_end_temp; i++)
            us_temp[i].binary_ = brain::binary();
        delete [] brain_.us;
        brain_.us = us_temp;
        us_temp = nullptr;
        brain_.quantity_of_neurons_in_bits = brainBits_;
        brain_.quantity_of_neurons = quantity_of_neuron_end_temp;
        brain_.quantity_of_neurons_binary = brain_.quantity_of_neurons - brain_.quantity_of_neurons_sensor - brain_.quantity_of_neurons_motor;
        brain_.reaction_rate = brain_.quantity_of_neurons;
    }
    brain_.mtx.unlock();
}
std::map<int, int> brain_friend::graphical_representation()
{
    std::vector<int> v;
    std::map<int, int> m;
    std::map<int, int>::iterator it;
    for(_word i = 0; i < brain_.quantity_of_neurons; i++)
        if(brain_.us[i].neuron_.get_type() == brain::neuron::neuron_type_binary)
            if(brain_.us[i].binary_.get_type_binary() == brain::binary::neuron_binary_type_in_work)
            {
                it = m.find(static_cast<int>(brain_.us[i].binary_.level));
                if (it == m.end())
                    m.insert(std::make_pair(static_cast<int>(brain_.us[i].binary_.level), 1));
                else
                    it->second++;
            }
    return m;
}
