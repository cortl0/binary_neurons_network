#
#   binary neurons network
#   created by Ilya Shishkin
#   cortl@8iter.ru
#   http://8iter.ru/ai.html
#   https://github.com/cortl0/binary_neurons_network
#   licensed by GPL v3.0
#

TARGET   = minimal
TEMPLATE = app

#QT       += core gui webenginewidgets

CONFIG += console c++17

SOURCES += \
        ../../brain/brain.cpp \
        ../../brain/m_sequence.cpp \
        ../../brain/neurons/binary.cpp \
        ../../brain/neurons/motor.cpp \
        ../../brain/neurons/neuron.cpp \
        ../../brain/neurons/sensor.cpp \
        ../../brain/random_put_get.cpp \
        ../../brain/thread.cpp \
        main.cpp

HEADERS += \
        ../../brain/brain.h \
        ../../brain/m_sequence.h \
        ../../brain/random_put_get.h \
        ../../brain/simple_math.h \
        ../../brain/config.h
