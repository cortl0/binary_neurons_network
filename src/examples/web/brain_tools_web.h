/*
 *   binary neurons network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef BRAIN_TOOLS_WEB_H
#define BRAIN_TOOLS_WEB_H

#include <fstream>

#include <QString>
#include <QFileDialog>
#include <QMessageBox>

#include "../../brain_tools.h"

namespace bnn
{

struct brain_tools_web : brain_tools
{
    QString brain_get_representation();
    QString brain_get_state();
    QString version = QString("0");
    std::map<int, int> graphical_representation();

    virtual ~brain_tools_web();
    brain_tools_web() = delete;
    brain_tools_web(u_word random_array_length_in_power_of_two,
                    u_word quantity_of_neurons_in_power_of_two,
                    u_word input_length,
                    u_word output_length,
                    u_word threads_count_in_power_of_two = 0);
    void load();
    void resize(u_word brainBits);
    void save();
    void stop();
};

} // namespace bnn

#endif // BRAIN_TOOLS_WEB_H
