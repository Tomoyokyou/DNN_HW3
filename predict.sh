FEATUREDIR=/home/ahpan/DNN_HW3/model/preprocess_3/
#FEATUREDIR=/home/hui/project/rnnFeat/
FEATUREFILE=${FEATUREDIR}word_vector.txt
SENTENCEFILE=${FEATUREDIR}training_oov.txt
CLASSFILE=${FEATUREDIR}classes.sorted.txt
TESTFILE=${FEATUREDIR}testing_data_parse2.txt
#FEATUREDIR=/home/hui/project/rnnFeat/
#FEATUREFILE=${FEATUREDIR}word_vector.txt
#SENTENCEFILE=${FEATUREDIR}training_oov.txt
#CLASSFILE=${FEATUREDIR}classes.sorted.txt
#TESTFILE=${FEATUREDIR}testing_data_parse2.txt
OUT=./result/out.csv
CUTCLASS=50
MODELFILE=./model/acc_31.mdl

mkdir -p result

if [ -f ./bin/predict.app ]; then
echo "executables checked..."
else
echo "missing exe:train.app..."
make predict
fi

 if [ -f ${MODELFILE} ]; then
	echo "Modelfile found"
./bin/predict.app  ${FEATUREFILE} ${SENTENCEFILE} ${CLASSFILE} ${TESTFILE} ${MODELFILE} --cutClass ${CUTCLASS} --outF ${OUT}
else
	echo "ERROR: Modelfile not found...(specify it in train.sh)"
 fi
