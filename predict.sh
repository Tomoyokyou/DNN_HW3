DATADIR=/home/hui/project/model/
TYPE=fbank/
TESTFILE=test/test_norm.svm
TESTARK=${DATADIR}${TYPE}test.ark
MAP=${DATADIR}phones/char_48to39.map
OUTCSV=out.csv
MODEL=model/new_single_e50.mdl

mkdir -p result

./bin/predict.app ${TESTFILE} ${MODEL} ${TESTARK} ${MAP} result/${OUTCSV}
echo "End of Testing!"
