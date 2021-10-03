#
#   binary neurons network
#   created by Ilya Shishkin
#   cortl@8iter.ru
#   https://github.com/cortl0/binary_neurons_network
#   licensed by GPL v3.0
#

TARGET = libbnnbrain

SRCS= \
brain.cpp \
m_sequence.cpp \
neurons/neuron.cpp \
neurons/binary.cpp \
neurons/motor.cpp \
neurons/sensor.cpp \
random_put_get.cpp \
storage.cpp \
thread.cpp \
../brain_friend.cpp

HDRS= \
brain.h \
config.h \
m_sequence.h \
neurons/neuron.h \
neurons/binary.h \
neurons/motor.h \
neurons/sensor.h \
random_put_get.h \
simple_math.h \
state.h \
storage.h \
thread.h \
../brain_friend.h

HDRS_RM= \
brain.h \
config.h \
m_sequence.h \
neurons/neuron.h \
neurons/binary.h \
neurons/motor.h \
neurons/sensor.h \
random_put_get.h \
simple_math.h \
state.h \
storage.h \
thread.h \
brain_friend.h

ifeq ($(PREFIX),)
	PREFIX := /usr/local
endif

PREFIX_INC=$(PREFIX)/include/bnn
PREFIX_LIB=$(PREFIX)/lib

.PHONY: all clean install uninstall

all: $(TARGET).so

$(TARGET).so: $(SRCS)
	g++ -std=c++17 -pthread -fPIC -shared -o $(TARGET).so $(SRCS)

clean:
	rm -f $(TARGET).so

install: $(TARGET).so
	install -d $(PREFIX_LIB)
	install -m 644 $(TARGET).so $(PREFIX_LIB)/$(TARGET).so
	install -d $(PREFIX_INC)
	install -t $(PREFIX_INC) $(HDRS)

uninstall:
	cd $(PREFIX_INC); rm $(HDRS_RM);
	cd $(PREFIX_LIB); rm $(TARGET).so;
