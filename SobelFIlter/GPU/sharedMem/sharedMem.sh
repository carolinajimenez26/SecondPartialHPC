#!/bin/bash

#SBATCH --job-name=sharedMem
#SBATCH --output=res_sharedMem.txt
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH  --gres=gpu:1


export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export CUDA_VISIBLE_DEVICES=0

./build/sharedMem.out ../../../images/image1.jpg

<<COMENT
for i in {1..10}
do
	for j in {1..20}
	do
		./build/sharedMem.out ../../../images/image$i.jpg >> times.txt
	done
	echo "Ready for image img$i.jpg"
done
COMENT
