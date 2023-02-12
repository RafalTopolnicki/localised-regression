#!/bin/bash

scriptname="2d_confidenceintervals.py"
mode="nearest"
method="BMR1"
X="U"

jobsfile="jobs_${X}_${method}.txt"
rm $jobsfile
for n in 500 1000 2000 3000
do
for X in U
do
    for eps in 0.01 0.1 0.3
    do
    echo "python ${scriptname} --method ${method} --n $n --a 1 --b 2 --c 0 --d 0 --e 0 --eps $eps --M 1000 --X $X --mode $mode" >> $jobsfile
    echo "python ${scriptname} --method ${method} --n $n --a 1 --b 2 --c -1 --d 3 --e 0 --eps $eps --M 1000 --X $X --mode $mode" >> $jobsfile
    echo "python ${scriptname} --method ${method} --n $n --a 1 --b 2 --c 0 --d 0 --e 0.2 --eps $eps --M 1000 --X $X --mode $mode" >> $jobsfile
    echo "python ${scriptname} --method ${method} --n $n --a 1 --b 2 --c 0 --d 0 --e 0.5 --eps $eps --M 1000 --X $X --mode $mode" >> $jobsfile
    echo "python ${scriptname} --method ${method} --n $n --a 1 --b 2 --c -1 --d 3 --e 0.5 --eps $eps --M 1000 --X $X --mode $mode" >> $jobsfile
    done
done
done
