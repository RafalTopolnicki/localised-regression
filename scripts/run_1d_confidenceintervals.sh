#!/bin/bash

scriptname="1d_confidenceintervals.py"
mode="nearest"

jobsfile="jobs_1d.txt"
rm $jobsfile
for n in 500 1000 2000 3000
do
  for method in LR BMR1 MARS BMR2 SVR
  do
  for X in U N
  do
      for eps in 0.1 0.3 1
      do
        echo "python ${scriptname} --method ${method} --n $n --a 2 --b 0 --c 0 --eps $eps --M 1000 --X $X --mode $mode" >> $jobsfile
        echo "python ${scriptname} --method ${method} --n $n --a 2 --b 1 --c 0 --eps $eps --M 1000 --X $X --mode $mode" >> $jobsfile
        echo "python ${scriptname} s--method ${method} --n $n --a 2 --b 0 --c 1 --eps $eps --M 1000 --X $X --mode $mode" >> $jobsfile
        echo "python ${scriptname} --method ${method} --n $n --a 0 --b 1 --c 0 --eps $eps --M 1000 --X $X --mode $mode" >> $jobsfile
        echo "python ${scriptname} --method ${method} --n $n --a 0 --b 0 --c 1 --eps $eps --M 1000 --X $X --mode $mode" >> $jobsfile
        echo "python ${scriptname} --method ${method} --n $n --a 2 --b 0.5 --c 0.25 --eps $eps --M 1000 --X $X --mode $mode" >> $jobsfile
        echo "python ${scriptname} --method ${method} --n $n --a 2 --b 0.5 --c -0.25 --eps $eps --M 1000 --X $X --mode $mode" >> $jobsfile
        echo "python ${scriptname} --method ${method} --n $n --a 2 --b -0.5 --c 0.25 --eps $eps --M 1000 --X $X --mode $mode" >> $jobsfile
        echo "python ${scriptname} --method ${method} --n $n --a 2 --b -0.5 --c -0.25 --eps $eps --M 1000 --X $X --mode $mode" >> $jobsfile
      done
  done
  done
done
parallel --jobs 14 < $jobsfile