DATADIR=/home/hui/model/
TYPE=fbank/
TESTFILE=${DATADIR}test/test_norm.svm
TESTARK=${DATADIR}test/test.ark
MAP=${DATADIR}char_48to39.map
OUTCSV=out.csv
MODEL=model/new_single_e50.mdl

mkdir -p result

./bin/predict.app ${TESTFILE} ${MODEL} ${TESTARK} ${MAP} result/${OUTCSV}
echo "End of Testing!"
