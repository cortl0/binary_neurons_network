/*
 *   binary neurons network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef BRAIN_FRIEND_WEB_H
#define BRAIN_FRIEND_WEB_H

#include <fstream>

#include <QString>
#include <QFileDialog>
#include <QMessageBox>

#include "../../brain_friend.h"

namespace bnn
{

struct brain_friend_web : brain_friend
{
    QString brain_get_representation();
    QString brain_get_state();
    QString version = QString("0");
    std::map<int, int> graphical_representation();

    virtual ~brain_friend_web();
    brain_friend_web() = delete;
    brain_friend_web(_word random_array_length_in_power_of_two,
                     _word quantity_of_neurons_in_power_of_two,
                     _word input_length,
                     _word output_length,
                     _word threads_count_in_power_of_two = 0);
    void load();
    void resize(_word brainBits);
    void save();
    void stop();
};

} // namespace bnn

#endif // BRAIN_FRIEND_WEB_H
