TARGET = brain.o
OBJ_DIR = ../brain_obj

.PHONY: all clean

all: $(TARGET)

clean:
	rm -rf $(OBJ_DIR)

$(TARGET): brain.cpp random_put_get.o
	@if (!(test -d $(OBJ_DIR))); then mkdir $(OBJ_DIR); fi
	gcc -c -std=c++11 -pthread -o $(OBJ_DIR)/$(TARGET) brain.cpp

random_put_get.o: random_put_get.cpp
	@if (!(test -d $(OBJ_DIR))); then mkdir $(OBJ_DIR); fi
	gcc -c -std=c++11 -o $(OBJ_DIR)/random_put_get.o random_put_get.cpp
