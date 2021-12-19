#
#   Binary Neurons Network
#   created by Ilya Shishkin
#   cortl@8iter.ru
#   http://8iter.ru/ai.html
#   https://github.com/cortl0/binary_neurons_network
#   licensed by GPL v3.0
#

TARGET   = minimal

TEMPLATE = app

CONFIG += console c++11

SOURCES += main.cpp

LIBS += /usr/local/lib/libbnn.so
