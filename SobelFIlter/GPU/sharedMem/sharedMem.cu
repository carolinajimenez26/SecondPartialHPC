#include<iostream>
#include<stdio.h>
#include<malloc.h>
#include<opencv2/opencv.hpp>
#include <time.h>
using namespace std;
using namespace cv;

#define RED 2
#define GREEN 1
#define BLUE 0
#define TILE_SIZE 32
#define MAX_MASK_WIDTH 9
#define MASK_WIDTH 3

__constant__ char M[MASK_WIDTH*MASK_WIDTH];

__device__
unsigned char clamp(int value){
  if (value < 0) value = 0;
  if (value > 255) value = 255;
  return (unsigned char)value;
}

//-------------------------------------------------------------------------------------------------------------
__global__ void sobelSharedMem(unsigned char *imageInput, int width, int height, unsigned int maskWidth,unsigned char *imageOutput){
    __shared__ float N_ds[TILE_SIZE + MASK_WIDTH - 1][TILE_SIZE+ MASK_WIDTH - 1];
    int n = maskWidth/2;
    int dest = threadIdx.y*TILE_SIZE+threadIdx.x, destY = dest / (TILE_SIZE+MASK_WIDTH-1), destX = dest % (TILE_SIZE+MASK_WIDTH-1),
        srcY = blockIdx.y * TILE_SIZE + destY - n, srcX = blockIdx.x * TILE_SIZE + destX - n,
        src = (srcY * width + srcX);
    if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
        N_ds[destY][destX] = imageInput[src];
    else
        N_ds[destY][destX] = 0;

    // Second batch loading
    dest = threadIdx.y * TILE_SIZE + threadIdx.x + TILE_SIZE * TILE_SIZE;
    destY = dest /(TILE_SIZE + MASK_WIDTH - 1), destX = dest % (TILE_SIZE + MASK_WIDTH - 1);
    srcY = blockIdx.y * TILE_SIZE + destY - n;
    srcX = blockIdx.x * TILE_SIZE + destX - n;
    src = (srcY * width + srcX);
    if (destY < TILE_SIZE + MASK_WIDTH - 1) {
        if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
            N_ds[destY][destX] = imageInput[src];
        else
            N_ds[destY][destX] = 0;
    }
    __syncthreads();

    int accum = 0;
    int y, x;
    for (y = 0; y < maskWidth; y++)
        for (x = 0; x < maskWidth; x++)
            accum += N_ds[threadIdx.y + y][threadIdx.x + x] * M[y * maskWidth + x];
    y = blockIdx.y * TILE_SIZE + threadIdx.y;
    x = blockIdx.x * TILE_SIZE + threadIdx.x;
    if (y < height && x < width)
        imageOutput[(y * width + x)] = clamp(accum);
    __syncthreads();
}

//--------------------------------------------------------------------------------------------------------------

__global__
void img2grayCU(unsigned char *imageInput, int width, int height, unsigned char *imageOutput){

  int row = blockIdx.y*blockDim.y+threadIdx.y;
  int col = blockIdx.x*blockDim.x+threadIdx.x;

  if((row < height) && (col < width)){

    imageOutput[row*width+col] = imageInput[(row*width+col)*3+RED]*0.299 + imageInput[(row*width+col)*3+GREEN]*0.587
    + imageInput[(row*width+col)*3+BLUE]*0.114;
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

  
  //-------------------- Masks -----------------------------

  int h_XMask[3*3] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
  int h_YMask[3*3] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
  char h_M[] = {1,0,-1,2,0,-2,1,0,-1};

//-------------------- copying to constant memory-------------------------

  error = cudaMemcpyToSymbol(M, h_M, sizeof(char)*MASK_WIDTH*MASK_WIDTH);
  if(error != cudaSuccess){
      printf("Error copying mask h_M to M\n");
      exit(-1);
  }
/*
  error = cudaMemcpyToSymbol(YM, h_YMask, sizeof(char)*MASK_WIDTH*MASK_WIDTH);
  if(error != cudaSuccess){
      printf("Error copying mask h_YMask to M\n");
      exit(-1);
  }
*/


  //------------------------ Sobel --------------------------------

  h_G = (unsigned char*)malloc(size);

  error = cudaMalloc((void**)&d_G, size);
  if (error != cudaSuccess) {
    printf("Error allocating memory for d_G\n");
    exit(-1);
  }

  // Convolution
  sobelSharedMem<<<dimGrid,dimBlock>>>(d_imageGray, width, height, MASK_WIDTH, d_G);
  cudaDeviceSynchronize();
/*
  // Convolution in Gy
  sobelSharedMem<<<dimGrid,dimBlock>>>(d_imageGray, width, height, MASK_WIDTH, d_Gy);
  cudaDeviceSynchronize();

  // Union of Gx and Gy results
  UnionCU<<<dimGrid,dimBlock>>>(d_G, d_Gx, d_Gy, height, width);
  cudaDeviceSynchronize();
 */

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

