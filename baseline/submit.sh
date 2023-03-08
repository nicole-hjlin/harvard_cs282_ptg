#!/bin/bash

set -x

read -p 'number of runs: ' n_runs
read -p 'job sh script name: ' file_name  

for((i=1; i<=n_runs; i++))
do 
	# echo $i
	sbatch $file_name
done