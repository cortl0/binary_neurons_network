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
        ../../brain/random_put_get.cpp \
        brain_friend.cpp \
        main.cpp \
        mainwindow.cpp \
        deviceai.cpp

HEADERS += \
        ../../brain/brain.h \
        ../../brain/random_put_get.h \
        ../../brain/simple_math.h \
        ../../brain/pch.h \
        brain_friend.h \
        deviceai.h \
        mainwindow.h

FORMS   += \
        mainwindow.ui
