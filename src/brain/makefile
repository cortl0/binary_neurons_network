#
#   binary neurons network
#   created by Ilya Shishkin
#   cortl@8iter.ru
#   https://github.com/cortl0/binary_neurons_network
#   licensed by GPL v3.0
#

TARGET = brain

SRCS=brain.cpp random_put_get.cpp m_sequence.cpp neurons/neuron.cpp neurons/binary.cpp neurons/motor.cpp neurons/sensor.cpp

ifeq ($(PREFIX),)
	PREFIX := /usr/local
endif

.PHONY: all clean install uninstall

all: $(TARGET).so

$(TARGET).so: $(SRCS)
	g++ -std=c++17 -pthread -fPIC -shared -o $(TARGET).so $(SRCS)

clean:
	rm -f $(TARGET).so

install: $(TARGET).so
	install -d $(PREFIX)/lib/
	install -m 644 $(TARGET).so $(PREFIX)/lib/$(TARGET).so
	install -d $(PREFIX)/include/
	install -m 644 $(TARGET).h $(PREFIX)/include/$(TARGET).h

uninstall:
	rm -f $(PREFIX)/lib/$(TARGET).so
	rm -f $(PREFIX)/include/$(TARGET).h
