#!/bin/bash
sed "s/^$//g" \
| sed "s///g" \
| sed  "/Project Gutenberg/,/\*END\*/d" \
| tr '\n' ' ' \
| sed "s/\t/ /g" \
| sed "s/\%//g" \
| sed "s/ [0-9][0-9]*/ num/g" \
| sed "s/ [^ ]*\.zip/ zp/g" \
| sed "s/ [^ ]*\.txt/ txt/g" \
| sed "s/\([a-zA-Z]\+\)'ll/\1 will/g" \
| sed "s/\([a-zA-Z]\+\)'ve/\1 have/g" \
| sed "s/n't/not/g" \
| sed "s/ \(He\|he\|She\|she\|It\|it\|How\|how\|what\|What\|Who\|who\|where\|Where\|That\|that\)'s / \1 is /g" \
| sed "s/^\(He\|he\|She\|she\|It\|it\|How\|how\|what\|What\|Who\|who\|where\|Where\|That\|that\)'s /\1 is /g" \
| sed "s/\([a-zA-Z][a-zA-Z]*\)'m /\1 am/g" \
| sed "s/\( \*\**[^\*][^\*]*\*\**\)/ \1\. /g" \
| sed "s/Mr\./Mr /g" \
| sed "s/Mrs\./Mrs /g" \
| sed "s/Ms\./Ms /g" \
| sed "s/St\./St /g" \
| sed "s/Dr\./Dr /g" \
| sed "s/(\([^()]*\))/\n\1\n/g" \
| sed "s/\"\([^\"]*\)\"/\n\1\n/g" \
| sed "s/\'\([^\']*\)\'/\n\1\n/g" \
| sed "s/\[[^][]*\]//g" \
| sed "s/[,]/ comma /g" \
| sed "s/[:\/']/ /g" \
| sed "s/[\?\!\.;]/\n/g" \
| sed "s/[^a-zA-Z0-9 ]/ /g" \
| sed "s/./\L&/g" \
| sed "s/wonot/will not/g" \
| sed "s/canot/can not/g" \
| sed "s/'d/ would/g" \
| sed "s/'in/ing/g" \
| sed "s/ ah / /g" \
| sed "s/\(is\|are\|am\|was\|were\|has\|have\|had\|will\|would\|could\|did\|do\|does\|should\|can\)not/\1 not/g" \
| sed "s/ \(a\|an\|the\) / articl /g" \
| sed "s/^\(a\|an\|the\) /articl /g" \
| sed "s/ [ ]*/ /g" \
| sed "/ comma $/d" \
| sed "/^ comma /d" \
| sed "s/^[ \t]*//g" \
| sed "s/[\t ]*$//g" \
| sed "/^$/d" \
| sed "/^[^ ]*$/d" \
| sed "s/[^[:print:]]//g" \
| sed "s/^/<s> /" \
| sed "s/$/ <\/s>/"
