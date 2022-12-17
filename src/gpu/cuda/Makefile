#
#   Binary Neurons Network
#   created by Ilya Shishkin
#   cortl@8iter.ru
#   http://8iter.ru/ai.html
#   https://github.com/cortl0/binary_neurons_network
#   licensed by GPL v3.0
#

include external/Makefile.cuda

# Target rules
all: build

project_name=bnn_gpu_cuda_test

BNN_INC_PATH=-I../..

build: $(project_name).out

check.deps:
ifeq ($(SAMPLE_ENABLED),0)
	@echo "Sample will be waived due to the above missing dependencies"
else
	@echo "Sample is ready - all dependencies have been met"
endif

$(project_name).cu: main.cpp
	ln -s main.cpp $(project_name).cu
	
$(project_name).o:$(project_name).cu
	$(EXEC) $(NVCC) $(INCLUDES) $(BNN_INC_PATH) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

$(project_name).out: $(project_name).o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

run: build
	$(EXEC) ./$(project_name).out

testrun: build
	$(EXEC) ./$(project_name).out --dummy-test-param

clean:
	rm $(project_name).out $(project_name).o $(project_name).cu

clobber: clean
