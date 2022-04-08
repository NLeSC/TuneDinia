#!/usr/bin/env bash

DIR="$( pwd )"
OCLDIR=$DIR/../
# echo $DIR
# echo $OCLDIR

bm="backprop bfs kmeans nw gaussian streamcluster hotspot leukocyte srad"

cd $OCLDIRz

for b in $bm; do
     cd $b
     python $b.py 
     cd $OCLDIR
done