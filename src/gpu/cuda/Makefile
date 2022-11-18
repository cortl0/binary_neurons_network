include external/Makefile.cuda

# Target rules
all: build

project_name=bnn_gpu_cuda_test

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
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

$(project_name).out: $(project_name).o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

run: build
	$(EXEC) ./$(project_name).out

testrun: build
	$(EXEC) ./$(project_name).out --dummy-test-param

clean:
	rm $(project_name).out $(project_name).o $(project_name).cu

clobber: clean
