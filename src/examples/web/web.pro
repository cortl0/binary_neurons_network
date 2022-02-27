#
#   Binary Neurons Network
#   created by Ilya Shishkin
#   cortl@8iter.ru
#   http://8iter.ru/ai.html
#   https://github.com/cortl0/binary_neurons_network
#   licensed by GPL v3.0
#

TARGET   = web
TEMPLATE = app
QT       += core gui webenginewidgets
CONFIG   += c++17
CONFIG -= app_bundle

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

DEFINES += QT_DEPRECATED_WARNINGS

LIBS += -lstdc++fs
LIBS += -pthread

SOURCES += \
        ../../brain/brain.cpp \
        ../../brain/m_sequence.cpp \
        ../../brain/neurons/binary.cpp \
        ../../brain/neurons/motor.cpp \
        ../../brain/neurons/neuron.cpp \
        ../../brain/neurons/sensor.cpp \
        ../../brain/random/random.cpp \
        ../../brain/thread.cpp \
        ../../brain_tools.cpp \
        brain_tools_web.cpp \
        deviceai.cpp \
        main.cpp \
        mainwindow.cpp

HEADERS += \
        ../../brain/brain.h \
        ../../brain/config.hpp \
        ../../brain/m_sequence.h \
        ../../brain/neurons/binary.h \
        ../../brain/neurons/motor.h \
        ../../brain/neurons/neuron.h \
        ../../brain/neurons/sensor.h \
        ../../brain/random/config.hpp \
        ../../brain/random/random.h \
        ../../brain/simple_math.hpp \
        ../../brain/state.hpp \
        ../../brain/storage.hpp \
        ../../brain/thread.h \
        ../../brain_tools.h \
        brain_tools_web.h \
        deviceai.h \
        mainwindow.h

FORMS   += \
        mainwindow.ui
