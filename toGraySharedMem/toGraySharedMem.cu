//Image to gray
#include<iostream>
#include<stdio.h>
#include<malloc.h>
//#include <cv.h>
//#include <highgui.h>
#include<opencv2/opencv.hpp>
using namespace std; 
using namespace cv;

#define RED 2
#define GREEN 1
#define BLUE 0
#define TILE_WIDTH 32*32


__device__
unsigned char clamp(int value){
    if(value < 0)
        value = 0;
    else
        if(value > 255)
            value = 255;
    return (unsigned char)value;
}

__global__ 
void img2gray(unsigned char *imageInput, int width, int height, unsigned char *imageOutput){

	// Matrix for save the data in shared memory
	int __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty threadIdx.y;

	// Identify the row and column of the d_P element to work on

    int row = by * TILE_WIDTH * ty;
    int col = bx * TILE_WIDTH * tx;

    int i, k;
    // Loop over the d_M and d_N tiles required to compute d_P element
    for(i = 0; i < width/TILE_WIDTH: ++i){

    	//Colaborative loading of d_M and d_N tiles into shared memory
    	Mds[ty][tx] = imageInput[row*width + i*TILE_WIDTH + tx];
    	__syncthreads();

    	
		imageOutput[row*width+col] = Mds[(row*width + i*TILE_WIDTH + tx)*3+RED]*0.299 + Mds[(row*width + i*TILE_WIDTH + tx)*3+GREEN]*0.587 
    	+ Mds[(row*width + i*TILE_WIDTH + tx)*3+BLUE]*0.114;
       }
       __syncthreads();
}

int main(int argc, char **argv){

	if(argc != 2){
		printf("Is required the name of the image with extension\n");
		return 1;
	}
    
    //For handle errors
    cudaError_t error = cudaSuccess;
    unsigned char *h_ImagenInicial, *d_ImagenInicial;
    //Image that will be pass to gray
	unsigned char *h_img_gray, *d_img_gray;
	char* imageName = argv[1];
	// Image readed
	Mat image;

	image = imread(imageName, 1);

	//Atributes of the image
	Size s = image.size();

	int width = s.width;
    int height = s.height;
    int sz = sizeof(unsigned char)*width*height*image.channels();
    // For the image in gray scale
	int size = sizeof(unsigned char)*width*height;

	// Separte memory for the intial image in host and device
	h_initialImagen = (unsigned char*)malloc(sz);
	error = cudaMalloc((void**)&d_initialImagen,sz);
    if(error != cudaSuccess){
        printf("Error asking memory in device for image\n");
        exit(-1);
	}

	// Pass the data to the readed image
	h_initialImagen = image.data;

	//Coṕy data to device
    error = cudaMemcpy(d_initialImagen,h_initialImagen,sz, cudaMemcpyHostToDevice);
    if(error != cudaSuccess){
        printf("Error copyng the data of h_ImagenInicial to d_ImagenInicial \n");
        exit(-1);
	}

	//Separate memory for gray images in host and device
    h_img_gray = (unsigned char*)malloc(size);
    
    error = cudaMalloc((void**)&d_img_gray,size);
    if(error != cudaSuccess){
        printf("Error asking memory for d_img_gray\n");
        exit(-1);
	}

	//Block of 32x32 threads = 1024 threads
	dim3 dimBlock(32,32,1);
    dim3 dimGrid(ceil(width/float(32)),ceil(height/float(32)),1); 
    img2gray<<<dimGrid,dimBlock>>>(d_initialImagen, width, height, d_img_gray);
	cudaDeviceSynchronize();

	//Copy data of gray image in device to host
    error = cudaMemcpy(h_img_gray,d_img_gray,size, cudaMemcpyDeviceToHost);
    if(error != cudaSuccess){
        printf("Error copyng data of d_img_gray to h_img_gray \n");
        exit(-1);
	}

	return 0;
}