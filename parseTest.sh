#!/bin/bash
sed "s/^$//g" \
| sed "s/^.*) //g" \
| sed "s/\[\(.*\)\]/\1/g" \
| sed "s/'ve/ have/g"  \
| sed "s/'m/ am/g" \
| sed "s/n't/not/g" \
| sed "s/ \(He\|he\|She\|she\|It\|it\|How\|how\|what\|What\|Who\|who\|where\|Where\|That\|that\)'s / \1 is /g" \
| sed "s/^\(He\|he\|She\|she\|It\|it\|How\|how\|what\|What\|Who\|who\|where\|Where\|That\|that\)'s /\1 is /g" \
| sed "s/'re/ are/g" \
| sed "s/'ll/ will/g" \
| sed "s/Mr\./Mr /g"  \
| sed "s/Mrs\./Mrs /g" \
| sed "s/Ms\./Ms /g" \
| sed "s/St\./St /g" \
| sed "s/Dr\./Dr /g" \
| sed "s/[,]/ comma /g" \
| sed "s/'/ /g" \
| sed "s/[\?\!\.;]//g" \
| sed "s/./\L&/g" \
| sed "s/wonot/will not/g" \
| sed "s/canot/can not/g" \
| sed "s/'d/ would/g" \
| sed "s/'in/ing/g" \
| sed "s/ ah / /g" \
| sed "s/\(is\|are\|am\|was\|were\|has\|have\|had\|will\|would\|could\|did\|do\|does\|should\|can\)not/\1 not/g" \
| sed "s/ \(a\|an\|the\) / articl /g" \
| sed "s/^\(a\|an\|the\) /articl /g" \
| sed "s/[ ]\+/ /g" \
| sed "s/^/<s> /g" \
| sed "s/$/ <\/s>/g"

