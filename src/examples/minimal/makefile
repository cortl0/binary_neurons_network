#
#   Binary Neurons Network
#   created by Ilya Shishkin
#   cortl@8iter.ru
#   https://github.com/cortl0/binary_neurons_network
#   licensed by GPL v3.0
#

TARGET = minimal.out

LIBBNN = -lbnn

SRCS = main.cpp

.PHONY: all clean

all: $(TARGET)

clean:
	rm -f $(TARGET)

$(TARGET): $(SRCS) $(LIBBNN)
	g++ -lpthread -o $(TARGET) $(SRCS) $(LIBBNN)
