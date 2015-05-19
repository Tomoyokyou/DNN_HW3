INPUT=/home/jason/training_pre4.txt
COUNT_OUTPUT=./model/training.cnt
LM_OUTPUT=./model/training4.lm

time ngram-count -text ${INPUT} -write ${COUNT_OUTPUT}
time ngram-count -read ${COUNT_OUTPUT} -lm ${LM_OUTPUT} -unk -order 4 
