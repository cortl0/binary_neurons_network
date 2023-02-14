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

#include "common/brain_tools.h"

namespace bnn
{

struct brain_tools_web : brain_tools
{
    QString brain_get_representation();
    QString brain_get_state();
    QString version = QString("0");
    std::map<u_word, u_word> graphical_representation();

    virtual ~brain_tools_web();
    brain_tools_web(const bnn_settings&);
    void load();
    void resize(u_word brainBits);
    void save();
    void stop();
};

} // namespace bnn

#endif // BRAIN_TOOLS_WEB_H
