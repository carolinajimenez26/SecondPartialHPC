#!/bin/bash

#SBATCH --job-name=Sobel
#SBATCH --output=Sobel.out
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH  --gres=gpu:1


export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export CUDA_VISIBLE_DEVICES=0

#./build/ImageSobel.out ../../../images/mec.png
for i in {1..10}
do
	for j in {1..20}
	do
		./build/ImageSobel.out ../../../images/image$i.jpg >> ./testTimes/test$i.csv
	done
	echo "Ready for image img$i.jpg"
done
