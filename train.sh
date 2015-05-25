FEATUREDIR=/home/ahpan/DNN_HW3/model/preprocess_3/
ANSDIR=/home/ahpan/Data/
#FEATUREDIR=/home/hui/project/rnnFeat/
#ANSDIR=/home/hui/project/rnnFeat/
FEATUREFILE=${FEATUREDIR}word_vector.txt
SENTENCEFILE=${FEATUREDIR}training_oov.txt
CLASSFILE=${FEATUREDIR}classes.sorted.txt
TESTFILE=${FEATUREDIR}testing_data_parse2.txt
ANSWERFILE=${ANSDIR}answer.txt

./bin/train.app  ${FEATUREFILE} ${SENTENCEFILE} ${CLASSFILE} ${TESTFILE}
