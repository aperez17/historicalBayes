#!/bin/bash
# Usage ./scripts/split_to_test.sh specificPreprocessed five-test-train

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage ./scripts/split_to_test.sh {inputDir} {outputDir}"
    exit 1
fi

inputDir=$1
outputDir=$2
trainDir="train"
testDir="test"
tmpDir="tempPreProcessed"
rm -rf $tmpDir
rm -rf $outputDir
mkdir -p $tmpDir
mkdir -p $outputDir
mkdir -p $outputDir/$trainDir
mkdir -p $outputDir/$testDir

cp -r $inputDir/* $tmpDir
for i in $( ls $tmpDir ); do
    mkdir -p $outputDir/$trainDir/$i
    mkdir -p $outputDir/$testDir/$i
    count=$(ls $tmpDir/$i | wc -l)
    iterations="$(($count/(10/5)))"
    echo "$count $iterations"
    for j in $( ls $tmpDir/$i ); do
        if [ $iterations == 0 ] ; then
            cp $tmpDir/$i/$j $outputDir/$trainDir/$i/$j
        else
            iterations=$(($iterations-1))
            cp $tmpDir/$i/$j $outputDir/$testDir/$i/$j
        fi
    done
done
