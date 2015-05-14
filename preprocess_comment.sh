#!/bin/bash
sed "s///g" \
| tr '\n' ' ' \
| sed "s/\t/ /g" \
| sed "s/.zip//g" \
| sed "s/.txt//g" \
| sed "s/(\([^()]*\))/\n\1\n/g" \
| sed "s/\"\([^\"]*\)\"/\n\1\n/g" \
| sed "s/\'\([^\']*\)\'/\n\1\n/g" \
| sed "s/\[[^][]*\]//g" \ 
| sed "s/[,:\/\` ]/ /g" \ # replace , : / ` as space
| sed "s/[\?\!\.;]/\n/g" \ # these ending character are replaced by \n
| sed "s/[^a-zA-Z0-9 ]/ /g" \ #other characters are replaced by space
| sed "s/./\L&/g" \ # don't know
| sed "s/ [ ]*/ /g" \ # if more than one space, stay one 
| sed "s/^[ \t]*//g" \ # delete tab and white space at front and end
| sed 's/[\t ]*$//g' \
| sed "/^$/d" \ # delete empty line
| sed '/^[^ ]*$/d' \ # delete lines with only one character
| sed "s/[^[:print:]]//g" \  #delete unprintable char
| sed "s/^/<s> /" \
| sed "s/$/ <\/s>/"
