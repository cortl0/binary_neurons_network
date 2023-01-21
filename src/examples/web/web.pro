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
LIBS += ../../build/cpu/libbnn-cpu.a

INCLUDEPATH += \
        ../..

SOURCES += \
        ../../common/sources/brain_tools.cpp \
        ../../cpu/cpu.cpp \
        brain_tools_web.cpp \
        deviceai.cpp \
        main.cpp \
        mainwindow.cpp

HEADERS += \
        common/headers/brain_tools.h \
        cpu/cpu.h \
        brain_tools_web.h \
        deviceai.h \
        mainwindow.h

FORMS   += \
        mainwindow.ui
