DATA=/home/hui/project/model/
DATA_LARRY=/home/larry/Documents/data/MLDS_HW1_RELEASE_v1/mfcc/
TYPE=fbank/
TRAIN=${DATA}train/train_norm.svm
TEST=${DATA}test/test_norm.svm
INDIM=69
OUTDIM=48
PHONUM=39
RATE=0.001
BSIZE=256
MAXEPOCH=100
DECAY=0.99999
DIM=${INDIM}-256-${OUTDIM}
OUTMODEL=model/new_single_e50.mdl

mkdir -p model

# *****************************************
# *********TRAINING FROM SCRATCH***********
# *****************************************

#valgrind --leak-check=yes 
gdb --args ./bin/train.app ${TRAIN} ${TEST} --rate ${RATE} --batchsize ${BSIZE} --maxEpoch ${MAXEPOCH} \
--momentum 0.9 --outName ${OUTMODEL} --decay ${DECAY} --variance 1.0 --dim ${DIM}

# ******************************************
# * THIS PART IS USED FOR LOADING DNN MODEL*
# ****************************************** 

#./bin/train.app ${TRAIN} ${TEST} ${LABEL} --rate ${RATE} --batchsize ${BSIZE} --maxEpoch ${MAXEPOCH} \
#--momentum 0.6 --outName model/single256_load.mdl --decay ${DECAY} --dim ${DIM} --load model/single256_load.mdl


echo "experiment done!"
