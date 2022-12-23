#!/bin/bash

FILE=Data/chiba/1111/2836_001.mov
XCAL=17.161340059327287 
YCAL=16.849643261459075

if [ ! -f $FILE ] ; then
    echo "no file: " $FILE
    exit
fi

poetry run python Patra2.py ${FILE} ${XCAL} ${YCAL}
