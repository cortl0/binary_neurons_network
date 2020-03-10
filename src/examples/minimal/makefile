TARGET = minimal_brain
RELEASE_DIR = ../minimal_release
BRAIN_DIR = ../../brain
BRAIN_OBJ_DIR = ../../brain_obj
PREFIX = /usr/local/bin

.PHONY: all clean install uninstall

all: $(TARGET)

clean:
	rm -rf $(RELEASE_DIR)
	cd $(BRAIN_DIR); make clean;

$(TARGET): main.o
	cd $(BRAIN_DIR); make all;
	@if (!(test -d $(RELEASE_DIR))); then mkdir $(RELEASE_DIR); fi
	g++ -std=c++11 -pthread -o $(RELEASE_DIR)/$(TARGET) $(RELEASE_DIR)/main.o $(BRAIN_OBJ_DIR)/brain.o $(BRAIN_OBJ_DIR)/random_put_get.o

main.o: main.cpp
	@if (!(test -d $(RELEASE_DIR))); then mkdir $(RELEASE_DIR); fi
	gcc -c -std=c++11 -o $(RELEASE_DIR)/main.o main.cpp

install:
	install $(RELEASE_DIR)/$(TARGET) $(PREFIX)

uninstall:
	rm -rf $(PREFIX)/$(TARGET)
