#!/bin/bash

scriptname="friedman.py"
mode="nearest"

jobsfile="jobs_friedman.txt"
rm $jobsfile
for n in 500 1000 2000
do
  for method in LR BMR1 BMR2 SVR SVRdef
  do
  for noise in 0.0 0.1 0.5 1.0 2.0
      do
        echo "python ${scriptname} --method ${method} --n ${n} --test_size 0.2 --type friedman1 --noise ${noise} --M 20" >> $jobsfile
        echo "python ${scriptname} --method ${method} --n ${n} --test_size 0.2 --type friedman2 --noise ${noise} --M 20" >> $jobsfile
        echo "python ${scriptname} --method ${method} --n ${n} --test_size 0.2 --type friedman3 --noise ${noise} --M 20" >> $jobsfile
      done
  done
done
parallel --jobs 64 < $jobsfile
