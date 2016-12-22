#!/bin/bash
# Usage ./scripts/update_jar.sh

currDir=$(pwd)
cd ClmetAnalytics
sbt clean compile assembly
cd $currDir
mkdir -p lib
cp ClmetAnalytics/target/scala-2.11/ClmetAnalytics-assembly-1.0.jar lib/ClemtAnalytics.jar
