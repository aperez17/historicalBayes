#!/bin/bash
# Usage ./scripts/pre_process_specific.sh clmet/corpus/txt/plain

if [ -z "$1" ] ; then
    echo "Usage ./scripts/pre_process_corpus.sh {inputDir}"
    exit 1
fi

inputDir=$1
tmpDir=tmp
output=preProcessedFive
rm -rf $output
rm -rf $tmpDir
mkdir -p $tmpDir
mkdir -p $tmpDir/runtime
mkdir -p $output/1710-1752
mkdir -p $output/1752-1794
mkdir -p $output/1794-1836
mkdir -p $output/1836-1878
mkdir -p $output/1878-1921

cp -r $inputDir/* $tmpDir

for i in $( ls $tmpDir ); do
    java -jar lib/ClemtAnalytics.jar -i $tmpDir/$i -s -t five -o $output
    if [ $? -eq 0 ]; then
        rm $tmpDir/$i
        echo "completed $i"
    else
        echo "FAILED TO PARSE $i"
    fi
done
