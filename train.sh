#FEATUREDIR=/home/ahpan/DNN_HW3/model/preprocess_3/
FEATUREDIR=/home/hui/project/rnnFeat/
FEATUREFILE=${FEATUREDIR}word_vector.txt
SENTENCEFILE=${FEATUREDIR}training_oov.txt
CLASSFILE=${FEATUREDIR}classes.sorted.txt
TESTFILE=${FEATUREDIR}testing_data_parse2.txt
#ANSDIR=/home/ahpan/Data/
#ANSDIR=/home/hui/project/rnnFeat/
#FEATUREDIR=model/
#ANSDIR=model/
#FEATUREFILE=${FEATUREDIR}word_vector.txt
#SENTENCEFILE=${FEATUREDIR}training_oov.txt
#CLASSFILE=${FEATUREDIR}classes.sorted.txt
#TESTFILE=${FEATUREDIR}testing_data_parse2.txt
#ANSWERFILE=${ANSDIR}answer.txt
RATE=0.01
MOMENTUM=0.8
EPOCH=100
DECAY=0.99
VAR=0.2
STEP=4
HIDDEN=50
HIDNUM=2
REG=0
OUT=./model/dRNN4.mdl
CUTCLASS=50

mkdir -p model
mkdir -p log

if [ -f ./bin/train.app ]; then
echo "executables checked..."
else
echo "missing exe:train.app..."
make train
fi
./bin/train.app  ${FEATUREFILE} ${SENTENCEFILE} ${CLASSFILE} ${TESTFILE} --ans ${ANSWERFILE} --rate ${RATE} \
--momentum ${MOMENTUM} --epoch ${EPOCH} --decay ${DECAY} --var ${VAR} --step ${STEP} --reg ${REG} \
 --hidden ${HIDDEN} --outF ${OUT} --cutClass ${CUTCLASS} --hidnum ${HIDNUM} | tee log/dRnn${STEP}.log
