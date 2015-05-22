INPUT=/home/ahpan/Data/train.txt
COUNT_OUTPUT=./model/training.cnt
LM_OUTPUT=./model/training.lm

time ngram-count -text ${INPUT} -write ${COUNT_OUTPUT}
time ngram-count -read ${COUNT_OUTPUT} -lm ${LM_OUTPUT} -unk -order 3
