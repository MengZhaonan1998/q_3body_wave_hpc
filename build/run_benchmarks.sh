#!/bin/bash

# run all benchmarks, defined in .slurm files

echo "COMMUNICATION BENCHMARKS:"
for processes in 128; do
	nodes=$(((processes-1)/48+1))
	if (( nodes > 1 )); then
		echo "With $processes processes, $nodes nodes:"
		sbatch -o slurm-$processes-$$.out -n $processes -N $nodes run_QRes.slurm  
	else
		echo "With $processes processes:"
		sbatch -o slurm-$processes-$$.out -n "$processes" run_QRes.slurm
	fi
done

#echo "POLYNOMIAL PRECONDITIONER BENCHMARKS:"
#for processes in 48; do
#	echo "With $processes processes:"
#	sbatch -o slurm-precond-$processes-$$.out -n "$processes" precond_benchmarks.slurm
#done

#echo "JACOBI VS GMRES BENCHMARKS:"
#for processes in 64 100 150; do
#	echo "With $processes processes:"
#	sbatch -o slurm-gmres-$processes-$$.out -n "$processes" gmres.slurm  
#        #sbatch -o slurm-jacobi-$processes-$$.out -n "$processes" jacobi.slurm  
#done
