#
#   binary neurons network
#   created by Ilya Shishkin
#   cortl@8iter.ru
#   https://github.com/cortl0/binary_neurons_network
#   licensed by GPL v3.0
#

TARGET = minimal_brain.out
BRAIN_DIR = ../../brain
SRCS_BRAIN=brain.cpp random_put_get.cpp m_sequence.cpp neurons/neuron.cpp neurons/binary.cpp neurons/motor.cpp neurons/sensor.cpp
OBJS_BRAIN=$(SRCS_BRAIN:.cpp=.o)
OBJS_BRAIN=$(SRCS_BRAIN:neurons/=)

SRCS=main.cpp $(BRAIN_DIR)/brain.o

.PHONY: all clean

all: $(TARGET)

clean:
	rm -f $(TARGET)

$(TARGET): $(SRCS)
	cd $(BRAIN_DIR); g++ -std=c++11 -pthread -c $(SRCS_BRAIN);
	cd $(BRAIN_DIR); g++ -std=c++11 -pthread -o ../examples/minimal/$(TARGET) ../examples/minimal/main.cpp $(OBJS_BRAIN);
