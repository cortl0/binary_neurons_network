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
    QString version = QString("0");
    QString brain_get_state();
    QString brain_get_representation();
    void save();
    void load();
    void stop();
    void resize(_word brainBits);
    brain_friend_web() = delete;
    brain_friend_web(bnn::brain &brain_);
    std::map<int, int> graphical_representation();
};

} // namespace bnn

#endif // BRAIN_FRIEND_WEB_H
