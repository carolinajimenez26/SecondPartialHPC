//Image to gray
#include<iostream>
#include<stdio.h>
#include<malloc.h>
#include <cuda.h>
#include <time.h>
//#include <cv.h>
//#include <highgui.h>
#include<opencv2/opencv.hpp>
using namespace std; 
using namespace cv;

#define RED 2
#define GREEN 1
#define BLUE 0

__device__
__host__
unsigned char clamp(int value){
    if(value < 0) value = 0;
    if(value > 255) value = 255;
    return (unsigned char)value;
}

__global__ 
void img2gray(unsigned char *imageInput, int width, int height, unsigned char *imageOutput){

    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    if((row < height) && (col < width)){
        imageOutput[row*width+col] = imageInput[(row*width+col)*3+RED]*0.299 + imageInput[(row*width+col)*3+GREEN]*0.587 + imageInput[(row*width+col)*3+BLUE]*0.114;
    }
}   

__global__
void convolution(){}

int main(int argc, char **argv){

    if(argc != 2){
  	printf("Is required the name of the image with extension\n");
	return 1;
    }
    
    //For handle errors
    cudaError_t error = cudaSuccess;
    unsigned char *h_initialImage, *d_initialImage;
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
    int sz = sizeof(unsigned char)*width*height*3;
    // For the image in gray scale
    int size = sizeof(unsigned char)*width*height;

    // Separte memory for the intial image in host and device
    h_initialImage = (unsigned char*)malloc(sz);
    error = cudaMalloc((void**)&d_initialImage,sz);
    if(error != cudaSuccess){
        printf("Error asking memory in device for image\n");
        exit(-1);
    }

    // Pass the data to the readed image
    h_initialImage = image.data;

    //Copy data to device
    error = cudaMemcpy(d_initialImage,h_initialImage,sz, cudaMemcpyHostToDevice);
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
    dim3 dimGrid(ceil(width/32.0),ceil(height/32.0),1); 
    img2gray<<<dimGrid,dimBlock>>>(d_initialImage, width, height, d_img_gray);
    cudaDeviceSynchronize();

   //Copy data of gray image in device to host
    error = cudaMemcpy(h_img_gray,d_img_gray,size, cudaMemcpyDeviceToHost);
    if(error != cudaSuccess){
        printf("Error copyng data of d_img_gray to h_img_gray \n");
        exit(-1);
    }
 
   //Copy the data of the h_img_gray to the Mat type for save
   Mat res_img_gray;
   res_img_gray.create(height, width, CV_8UC1);
   res_img_gray.data = h_img_gray;
   imwrite("gray_image.png", res_img_gray);
     

   free(h_img_gray);
   free(h_initialImage);
   cudaFree(d_img_gray);
   cudaFree(d_initialImage);    

    return 0;
}
