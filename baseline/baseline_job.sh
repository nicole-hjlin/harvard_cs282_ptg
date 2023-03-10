#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=1G      
#SBATCH -t 0-02:00         
#SBATCH -p seas_compute,shared,serial_requeue,tambe
#SBATCH -o /n/home10/hongjinlin/outputs/cs282_ptg_baseline_%j.out
#SBATCH -e /n/home10/hongjinlin/outputs/cs282_ptg_baseline_%j.err 

set -x

module load python/3.9.12-fasrc01
module load GCC/8.2.0-2.31.1

python main.py --experiment loo_200 --n 10 --loo
python main.py --experiment rs_200 --n 10 