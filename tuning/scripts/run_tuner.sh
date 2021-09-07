#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OCLDIR=$DIR/../

# 10 in total
#  and  does not work
bm="backprop b+tree dwt2d heartwall hotspot3D kmeans hybridsort \
    leukocyte myocyte \
    nw pathfinder streamcluster bfs cfd gaussian hotspot \
    lavaMD lud nn particlefilter srad"

bm="backprop bfs gaussian hotspot kmeans"

OUTDIR=$DIR/results-gpu
mkdir $OUTDIR &>/dev/null

echo $bm

cd $OCLDIR
exe() { echo "++ $@" |& tee -a $OUTDIR/$b.txt ; \
    $@ |& tee -a $OUTDIR/$b.txt ; }

for b in $bm; do
    echo "$(date) # running $b"
    cd $b
    srun --gres=gpu:1 python $b.py

    cd $OCLDIR

done