#!/bin/bash

FILE=sample/Ba_dance_3.mov
XCAL=15.479876160990711
YCAL=15.598232784202677

if [ ! -f $FILE ] ; then
    echo "no file: " $FILE
    exit
fi

echo "Attention: clibration data is a dummy sample"
poetry run python Patra2.py ${FILE} ${XCAL} ${YCAL}
