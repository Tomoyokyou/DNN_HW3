INPUT=/home/jason/training_pre4.txt
W2V=/home/larry/Documents/MLDS/trunk/word2vec
DISTANCE=
OUTPUT=./model/word_vector.txt
CLASS=./model/classes.txt
SORTEDCLASS=./model/classes.sorted.txt

time /home/larry/Documents/MLDS/trunk/word2vec -train ${INPUT} -output ${OUTPUT} -cbow 1 -size 200 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 20 -binary 0 -iter 15
#./distance vectors.bin


time /home/larry/Documents/MLDS/trunk/word2vec -train ${INPUT} -output ${CLASS} -cbow 1 -size 200 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 20 -iter 15 -classes 500
sort ${CLASS} -k 2 -n > ${SORTEDCLASS}
