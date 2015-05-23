CC=gcc
CXX=g++
CPPFLAGS= -O3 -std=c++11
NVCC=nvcc -arch=sm_21 -w
CUDA_DIR=/usr/local/cuda/
EIGENDIR=/usr/local/include/eigen3/

EXECUTABLES=#train predict
LIBCUMATDIR=tool/libcumatrix/
CUMATOBJ=$(LIBCUMATDIR)obj/device_matrix.o $(LIBCUMATDIR)obj/cuda_memory_manager.o
HEADEROBJ=obj/util.o obj/transforms.o obj/rnn.o obj/dataset.o obj/parser.o

LIBS=$(LIBCUMATDIR)lib/libcumatrix.a
# +==============================+
# +======== Phony Rules =========+
# +==============================+

.PHONY: debug all clean 

all:DIR TOOL $(HEADEROBJ) $(EXECUTABLES)

debug:CPPFLAGS+=-g -DDEBUG


vpath %.h include/
vpath %.cpp src/
vpath %.cu src/

INCLUDE= -I include/\
	 -I $(LIBCUMATDIR)include/\
	 -I $(CUDA_DIR)include/\
	 -I $(CUDA_DIR)samples/common/inc/\
	 -I $(EIGENDIR)

LD_LIBRARY=-L$(CUDA_DIR)lib64 -L$(LIBCUMATDIR)lib
LIBRARY=-lcuda -lcublas -lcudart -lcumatrix

TOOL:
	@echo "Checking library file in tool/libcumatrix"
	@if test -f $(LIBS); then \
		echo "Library file found"; \
	else \
		echo "Missing library file...regenerating"; \
		cd tool/libcumatrix/ ;  make ; cd ../.. ; \
	fi
DIR:
	@echo "checking object and executable directory..."
	@mkdir -p obj
	@mkdir -p bin

larry:$(HEADEROBJ) example/classifyWords.cpp
	@echo "compiling testViterbi.cpp"
	@$(CXX) $(CPPFLAGS) $(INCLUDE) -o bin/$@.app $^ $(LIBS) $(LIBRARY) $(LD_LIBRARY)

train:$(HEADEROBJ) example/train.cpp
	@echo "compiling train.app for DNN Training"
	@$(CXX) $(CPPFLAGS) $(INCLUDE) -o bin/$@.app $^ $(LIBS) $(LIBRARY) $(LD_LIBRARY)

predict:$(HEADEROBJ) example/predict.cpp
	@echo "compiling predict.app for DNN Testing"
	@$(CXX) $(CPPFLAGS) $(INCLUDE) -o bin/$@.app $^ $(LIBS) $(LIBRARY) $(LD_LIBRARY)

jason:$(HEADEROBJ) example/dataTest.cpp
	@echo "compiling dataTest.app for Dataset Testing"
	@$(CXX) $(CPPFLAGS) $(INCLUDE) -o bin/$@.app $^ $(LIBS) $(LIBRARY) $(LD_LIBRARY)
clean:
	@echo "All objects and executables removed"
	@rm -f $(EXECUTABLES) obj/* bin/*.app

cleanOrig:
	@echo "All backup file *.orig removed"
	@find . -name "*.orig" -type f
	@find . -name "*.orig" -type f -delete

ctags:
	@rm -f src/tags tags
	@echo "Tagging src directory"
	@cd src; ctags -a ./* ../include/*.h ; cd ..
	@echo "Tagging example directory"
	@cd example; ctags -a ./* ../include/*.h ../src/* ; cd ..
	@echo "Tagging main directory"
	@ctags -a example/* src/* include/*.h ./*
	
# +==============================+
# +===== Other Phony Target =====+
# +==============================+
obj/%.o: src/%.cpp include/%.h
	@echo "compiling OBJ: $@ " 
	@$(CXX) $(CPPFLAGS) $(INCLUDE) -o $@ -c $<

obj/%.o: %.cu
	@echo "compiling OBJ: $@ "
	@$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) $(INCLUDE) -o $@ -c $<
