# //*************************************************************//
# //                                                             //
# //   binary neurons network                                    //
# //   created by Ilya Shishkin                                  //
# //   cortl@8iter.ru                                            //
# //   http://8iter.ru/ai.html                                   //
# //   https://github.com/cortl0/binary_neurons_network          //
# //   licensed by GPL v3.0                                      //
# //                                                             //
# //*************************************************************//

TARGET   = web
TEMPLATE = app
QT       += core gui webenginewidgets
CONFIG   += c++11

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

DEFINES += QT_DEPRECATED_WARNINGS

SOURCES += \
        ../../brain/brain.cpp \
        ../../brain/m_sequence.cpp \
        ../../brain/neurons/binary.cpp \
        ../../brain/neurons/motor.cpp \
        ../../brain/neurons/neuron.cpp \
        ../../brain/neurons/sensor.cpp \
        ../../brain/random_put_get.cpp \
        brain_friend.cpp \
        deviceai.cpp \
        main.cpp \
        mainwindow.cpp

HEADERS += \
        ../../brain/brain.h \
        ../../brain/m_sequence.h \
        ../../brain/random_put_get.h \
        ../../brain/simple_math.h \
        ../../brain/config.h \
        brain_friend.h \
        deviceai.h \
        mainwindow.h

FORMS   += \
        mainwindow.ui
