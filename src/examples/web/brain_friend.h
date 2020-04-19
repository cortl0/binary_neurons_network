//*************************************************************//
//                                                             //
//   binary neurons network                                    //
//   created by Ilya Shishkin                                  //
//   cortl@8iter.ru                                            //
//   http://8iter.ru/ai.html                                   //
//   https://github.com/cortl0/binary_neurons_network          //
//   licensed by GPL v3.0                                      //
//                                                             //
//*************************************************************//

#ifndef BRAIN_FRIEND_H
#define BRAIN_FRIEND_H

#include <QString>
#include <QFileDialog>
#include <QMessageBox>

#include "../../brain/brain.h"

struct brain_friend
{
    brain &brain_;
    QString version = QString("0");
    QString brain_get_state();
    QString brain_get_representation();
    void save();
    void load();
    void stop();
    void resize(_word brainBits);
    brain_friend() = delete;
    brain_friend(brain &brain_) : brain_(brain_) {}
    std::map<int, int> graphical_representation();
};

#endif // !BRAIN_FRIEND_H
