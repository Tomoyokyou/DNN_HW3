CC=gcc
CXX=g++
CPPFLAGS=  -O3 -std=c++11
EIGENDIR=/usr/local/include/eigen3/

EXECUTABLES=train calAcc#predict
HEADEROBJ=obj/util.o obj/transforms.o obj/rnn.o obj/dataset.o obj/parser.o

LIBS=#$(LIBCUMATDIR)lib/libcumatrix.a
# +==============================+
# +======== Phony Rules =========+
# +==============================+

.PHONY: debug all clean 

all:DIR $(HEADEROBJ) $(EXECUTABLES)

debug:CPPFLAGS+=-g -DDEBUG


vpath %.h include/
vpath %.cpp src/
#vpath %.cu src/

INCLUDE= -I include/\
	 -I $(EIGENDIR)

LD_LIBRARY=#-L$(CUDA_DIR)lib64 -L$(LIBCUMATDIR)lib
LIBRARY=#-lcuda -lcublas -lcudart -lcumatrix

DIR:
	@echo "checking object and executable directory..."
	@mkdir -p obj
	@mkdir -p bin

#larry:$(HEADEROBJ) example/testLoadModel.cpp
#	@echo "compiling testLoadModel.cpp"
#	@$(CXX) $(CPPFLAGS) $(INCLUDE) -o bin/$@.app $^ $(LIBS) $(LIBRARY) $(LD_LIBRARY)

train:$(HEADEROBJ) example/rnnTrain.cpp
	@echo "compiling train.app for DNN Training"
	@$(CXX) $(CPPFLAGS) $(INCLUDE) -o bin/$@.app $^ $(LIBS) $(LIBRARY) $(LD_LIBRARY)

#predict:$(HEADEROBJ) example/predict.cpp
#	@echo "compiling predict.app for DNN Testing"
#	@$(CXX) $(CPPFLAGS) $(INCLUDE) -o bin/$@.app $^ $(LIBS) $(LIBRARY) $(LD_LIBRARY)

#jason:$(HEADEROBJ) example/dataTest.cpp
#	@echo "compiling dataTest.app for Dataset Testing"
#	@$(CXX) $(CPPFLAGS) $(INCLUDE) -o bin/$@.app $^ $(LIBS) $(LIBRARY) $(LD_LIBRARY)
calAcc: example/calAcc.cpp
	@echo "compiling calAcc.app to calculate accuracy"
	@$(CXX) ${CPPFLAGS} -o bin/calAcc.app $^
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

