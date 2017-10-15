#include<iostream>
#include<stdio.h>
#include<malloc.h>
#include<opencv2/opencv.hpp>
// #include "timer.cpp"
#include <time.h>
using namespace std;
using namespace cv;

#define RED 2
#define GREEN 1
#define BLUE 0


__device__
__host__
unsigned char clamp(int value){
  if (value < 0) value = 0;
  if (value > 255) value = 255;
  return (unsigned char)value;
}

__host__
void print(unsigned char *M, int rows, int cols){
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      printf("%d ", M[(i * cols) + j]);
    }
    printf("\n");
  }
}

__host__
void convolution(unsigned char *imageInput, int mask[3][3], int rows, int cols, unsigned char *imageOutput){

  for(int i = 0; i < rows; i++) {
    for(int j = 0; j < cols; j++) {
      int sum = 0;
      int aux_cols = j - 1, aux_rows = i - 1;

      for(int k = 0; k < 3; k++) { //mask_rows
        for(int l = 0; l < 3; l++) { //mask_cols
          if ((aux_rows >= 0 && aux_cols >= 0) && (aux_rows < rows && aux_cols < cols))

          sum += mask[k][l]*imageInput[(aux_rows*cols)+ aux_cols];

          aux_cols++;
        }
        aux_rows++;
        aux_cols = j - 1;
      }

      imageOutput[(i * cols) + j] = clamp(sum);
    }
  }
}

__global__
void convolutionCU(unsigned char *imageInput, int *mask, int rows, int cols, unsigned char *imageOutput){

  int i = blockIdx.y*blockDim.y+threadIdx.y;
  int j = blockIdx.x*blockDim.x+threadIdx.x;
  int sum = 0;

  if (i < rows && j < cols) {

    int aux_cols = j - 1, aux_rows = i - 1;
    for (int k = 0; k < 3; k++) {//mask_rows
      for (int l = 0; l < 3; l++) {//mask_cols
        if(aux_rows >= 0 && aux_cols >= 0 && aux_rows < rows && aux_cols < cols)
        sum += mask[(k*3) + l] * imageInput[(aux_rows*cols) + aux_cols];

        aux_cols++;
      }
      aux_rows++;
      aux_cols = j - 1;
    }
    imageOutput[(i * cols) + j] = clamp(sum);
  }
}


__host__
void img2gray(unsigned char *imageInput, int width, int height, unsigned char *imageOutput){

  for(int row = 0; row < height; row++){
    for(int col = 0; col < width; col++){
      imageOutput[row*width+col] = imageInput[(row*width+col)*3+RED]*0.299 + imageInput[(row*width+col)*3+GREEN]*0.587 + imageInput[(row*width+col)*3+BLUE]*0.114;
    }
  }
}


__global__
void img2grayCU(unsigned char *imageInput, int width, int height, unsigned char *imageOutput){

  int row = blockIdx.y*blockDim.y+threadIdx.y;
  int col = blockIdx.x*blockDim.x+threadIdx.x;

  if((row < height) && (col < width)){

    imageOutput[row*width+col] = imageInput[(row*width+col)*3+RED]*0.299 + imageInput[(row*width+col)*3+GREEN]*0.587
    + imageInput[(row*width+col)*3+BLUE]*0.114;
  }
}


__host__
void Union(unsigned char *imageOutput, unsigned char *Gx, unsigned char *Gy, int rows, int cols){
  for(int i = 0; i < rows; i++){
    for(int j = 0; j < cols; j++){
      imageOutput[(i * cols) + j] = sqrt(pow(Gx[(i * cols) + j],2) + pow(Gx[(i * cols) + j],2));
    }
  }
}


__global__
void UnionCU(unsigned char *imageOutput, unsigned char *Gx, unsigned char *Gy, int rows, int cols){

  int i = blockIdx.y*blockDim.y+threadIdx.y;
  int j = blockIdx.x*blockDim.x+threadIdx.x;

  if (i < rows && j < cols){
    imageOutput[(i * cols) + j] = sqrtf((Gx[(i * cols) + j] * Gx[(i * cols) + j]) + (Gy[(i * cols) + j] * Gy[(i * cols) + j]) );
  }
}


void write(Size s, char* fileName, double elapsedTime){
  long size = s.width * s.height;
  FILE *f = fopen("../global.time", "a");
  if (f == NULL) printf("Error opening file!\n");
  else {
    fprintf(f, "%ld %s %lf\n", size, fileName, elapsedTime);
  }
  fclose(f);
}

int main(int argc, char **argv){

  cudaError_t error = cudaSuccess;
  clock_t start, end;
  unsigned char *h_imageInput, *d_imageInput, *h_imageGray, *d_imageGray;
  unsigned char *d_Gx, *d_Gy, *h_G, *d_G; // Sobel Operators
  int *d_XMask, *d_YMask;
  char* imageName = argv[1];
  Mat image;

  if (argc != 2) {
    printf("Usage: Image path\n");
    return 1;
  }

  image = imread(imageName, 1);

  if (!image.data) {
    printf("No image Data\n");
    return 1;
  }

  // imshow("Image input", image);
  // waitKey(0);
  //
  // // ------------------------- Gray ------------------------------

  // Timer t("Sobel_Global");
  start = clock();

  Size s = image.size();

  int width = s.width;
  int height = s.height;
  int sz = sizeof(unsigned char) * width * height * image.channels();
  int size = sizeof(unsigned char) * width * height;


  h_imageInput = (unsigned char*)malloc(sz);

  error = cudaMalloc((void**)&d_imageInput,sz);
  if (error != cudaSuccess) {
    printf("Error allocating memory for d_imageInput\n");
    exit(-1);
  }

  h_imageInput = image.data;

  error = cudaMemcpy(d_imageInput, h_imageInput, sz, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    printf("Error copying data from h_imageInput to d_imageInput\n");
    exit(-1);
  }

  h_imageGray = (unsigned char*)malloc(size);

  error = cudaMalloc((void**)&d_imageGray, size);
  if (error != cudaSuccess) {
    printf("Error allocating memory for d_imageGray\n");
    exit(-1);
  }

  int blockSize = 32;
  dim3 dimBlock(blockSize, blockSize, 1);
  dim3 dimGrid(ceil(width/float(blockSize)), ceil(height/float(blockSize)), 1);
  img2grayCU<<<dimGrid,dimBlock>>>(d_imageInput, width, height, d_imageGray);
  cudaDeviceSynchronize();

  error = cudaMemcpy(h_imageGray, d_imageGray, size, cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    printf("Error copying data from d_imageGray to h_imageGray\n");
    exit(-1);
  }

/*
  Mat result_imageGray;
  result_imageGray.create(height, width, CV_8UC1);
  result_imageGray.data = h_imageGray;
*/
  // imshow("Gray image CUDA", result_imageGray);
  // waitKey(0);
  // imwrite("Gray_image_CUDA.jpg", result_imageGray);


  //-------------------- Masks -----------------------------

  error = cudaMalloc((void**)&d_XMask, 3*3*sizeof(int));
  if (error != cudaSuccess) {
    printf("Error allocating memory for d_XMask\n");
    exit(-1);
  }

  error = cudaMalloc((void**)&d_YMask, 3*3*sizeof(int));
  if (error != cudaSuccess) {
    printf("Error reservando memoria para d_Mascara_Y\n");
    exit(-1);
  }

  int h_XMask[3*3] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
  int h_YMask[3*3] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

  error = cudaMemcpy(d_XMask, h_XMask, 3*3*sizeof(int), cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    printf("Error copying data from h_XMask to d_XMask\n");
    exit(-1);
  }

  error = cudaMemcpy(d_YMask, h_YMask, 3*3*sizeof(int), cudaMemcpyHostToDevice);
  if(error != cudaSuccess){
    printf("Error copying data from h_YMask to d_YMask\n");
    exit(-1);
  }

  //------------------------ Sobel --------------------------------

  h_G = (unsigned char*)malloc(size);

  error = cudaMalloc((void**)&d_G, size);
  if (error != cudaSuccess) {
    printf("Error allocating memory for d_G\n");
    exit(-1);
  }

  error = cudaMalloc((void**)&d_Gx, size);
  if (error != cudaSuccess) {
    printf("Error allocating memory for d_Gx\n");
    exit(-1);
  }

  error = cudaMalloc((void**)&d_Gy, size);
  if (error != cudaSuccess) {
    printf("Error allocating memory for d_Gy\n");
    exit(-1);
  }

  // Convolution in Gx
  convolutionCU<<<dimGrid,dimBlock>>>(d_imageGray, d_XMask, height, width, d_Gx);
  cudaDeviceSynchronize();

  // Convolution in Gy
  convolutionCU<<<dimGrid,dimBlock>>>(d_imageGray, d_YMask, height, width, d_Gy);
  cudaDeviceSynchronize();

  // Union of Gx and Gy results
  UnionCU<<<dimGrid,dimBlock>>>(d_G, d_Gx, d_Gy, height, width);
  cudaDeviceSynchronize();

  error = cudaMemcpy(h_G, d_G, size, cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    printf("Error copying data from d_G to h_G\n");
    exit(-1);
  }

  Mat result_Sobel;
  result_Sobel.create(height, width, CV_8UC1);
  result_Sobel.data = h_G;

  // imshow("Sobel CUDA", result_Sobel);
  // waitKey(0);
  imwrite("Sobel_Shared.png", result_Sobel);

  // write(s, imageName, t.elapsed());
  end = clock();
  double time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  // printf("elapsed time: %lf", time_used);
  write(s, imageName, time_used);

  // free(h_imageInput);
  cudaFree(d_imageInput);
  free(h_imageGray);
  cudaFree(d_imageGray);
  cudaFree(d_XMask);
  cudaFree(d_YMask);
  free(h_G);
  cudaFree(d_Gx);
  cudaFree(d_Gy);
  cudaFree(d_G);

  return 0;
}

