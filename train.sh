FEATUREDIR=/home/ahpan/DNN_HW3/model/preprocess_3/                                                                                                        
ANSDIR=/home/ahpan/Data/
#FEATUREDIR=/home/hui/project/rnnFeat/
#ANSDIR=/home/hui/project/rnnFeat/
FEATUREFILE=${FEATUREDIR}word_vector.txt
SENTENCEFILE=${FEATUREDIR}training_oov.txt
CLASSFILE=${FEATUREDIR}classes.sorted.txt
TESTFILE=${FEATUREDIR}testing_data_parse2.txt
#ANSWERFILE=${ANSDIR}answer.txt
RATE=0.01
MOMENTUM=0
EPOCH=1
DECAY=0.99
VAR=0.2
STEP=5
HIDDEN=50
REG=0
OUT=./model/out.mdl
CUTCLASS=50

mkdir -p model

if [ -f ./bin/train.app ]; then
echo "executables checked..."
else
echo "missing exe:train.app..."
make train
fi
./bin/train.app  ${FEATUREFILE} ${SENTENCEFILE} ${CLASSFILE} ${TESTFILE} --ans ${ANSWERFILE} --rate ${RATE} \
--momentum ${MOMENTUM} --epoch ${EPOCH} --decay ${DECAY} --var ${VAR} --step ${STEP} --reg ${REG} \
 --hidden ${HIDDEN} --outF ${OUT} --cutClass ${CUTCLASS}
