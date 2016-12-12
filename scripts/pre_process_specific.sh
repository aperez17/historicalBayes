#!/bin/bash
# Usage ./scripts/pre_process_specific.sh clmet/corpus/txt/plain

if [ -z "$1" ] ; then
    echo "Usage ./scripts/pre_process_corpus.sh {inputDir}"
    exit 1
fi

inputDir=$1
tmpDir=tmp
output=preProcessedSeven
rm -rf $output
rm -rf $tmpDir
mkdir -p $tmpDir
mkdir -p $tmpDir/runtime
mkdir -p $output/1710-1740
mkdir -p $output/1740-1770
mkdir -p $output/1770-1800
mkdir -p $output/1800-1830
mkdir -p $output/1830-1860
mkdir -p $output/1860-1890
mkdir -p $output/1890-1921

cp -r $inputDir/* $tmpDir

for i in $( ls $tmpDir ); do
    java -jar lib/ClemtAnalytics.jar -i $tmpDir/$i -s -t seven -o $output
    if [ $? -eq 0 ]; then
        rm $tmpDir/$i
        echo "completed $i"
    else
        echo "FAILED TO PARSE $i"
    fi
done
