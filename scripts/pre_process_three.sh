#!/bin/bash
# Usage ./scripts/pre_process_three.sh clmet/corpus/txt/plain

if [ -z "$1" ] ; then
    echo "Usage ./scripts/pre_process_three.sh {inputDir}"
    exit 1
fi

inputDir=$1
tmpDir=tmp
output=preProcessedCorpus
rm -rf $tmpDir
mkdir -p $tmpDir
mkdir -p $tmpDir/runtime
mkdir -p $output/1710-1780
mkdir -p $output/1780-1850
mkdir -p $output/1850-1920

cp -r $inputDir/* $tmpDir

for i in $( ls $tmpDir ); do
    java -jar lib/ClemtAnalytics.jar -i $tmpDir/$i -s -o $output
    if [ $? -eq 0 ]; then
        rm $tmpDir/$i
        echo "completed $i"
    else
        echo "FAILED TO PARSE $i"
    fi
done
