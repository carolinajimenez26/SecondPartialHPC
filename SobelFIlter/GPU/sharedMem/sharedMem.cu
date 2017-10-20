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
//__constant__ char YM[MASK_WIDTH*MASK_WIDTH];

__device__
unsigned char clamp(int value){
  if (value < 0) value = 0;
  if (value > 255) value = 255;
  return (unsigned char)value;
}

//-------------------------------------------------------------------------------------------------------------
/*__global__ void sobelSharedMem(unsigned char *imageInput, int width, int height, unsigned int maskWidth,unsigned char *imageOutput){
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

*/

__global__ void sobelSharedMem(unsigned char *imageInput, int width, int height, \
        unsigned int maskWidth,unsigned char *imageOutput){

    int size = TILE_SIZE + MASK_WIDTH - 1;
    //__shared__ float N_ds[size][size];
    __shared__ float N_ds[TILE_SIZE + MASK_WIDTH - 1][TILE_SIZE+ MASK_WIDTH - 1];
    int n = maskWidth/2;
    int dest = threadIdx.y*TILE_SIZE+threadIdx.x, destY = dest / size, destX = dest % size,
        srcY = blockIdx.y * TILE_SIZE + destY - n, srcX = blockIdx.x * TILE_SIZE + destX - n,
        src = (srcY * width + srcX);
    if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
        N_ds[destY][destX] = imageInput[src];
    else
        N_ds[destY][destX] = 0;

    // Second batch loading
    dest = threadIdx.y * TILE_SIZE + threadIdx.x + TILE_SIZE * TILE_SIZE;
    destY = dest / size, destX = dest % size;
    srcY = blockIdx.y * TILE_SIZE + destY - n;
    srcX = blockIdx.x * TILE_SIZE + destX - n;
    src = (srcY * width + srcX);
    if (destY < size) {
        if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
            N_ds[destY][destX] = imageInput[src];
        else
            N_ds[destY][destX] = 0;
    }
    __syncthreads();

    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;

    if (x > height || y > width)
      return;

    int cur = 0, nx, ny;
    for (int i = 0; i < maskWidth; ++i) {
      for (int j = 0; j < maskWidth; ++j) {
        nx = threadIdx.y + i;
        ny = threadIdx.x + j;
        if (nx >= 0 && nx < size && ny >= 0 && ny < size) {
          cur += N_ds[nx][ny] * M[i * maskWidth + j];
        }
      }
    }
    imageOutput[x * width + y] = clamp(cur);
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

int main(int argc, char **argv){

  cudaError_t error = cudaSuccess;
  clock_t start, end;
  double time_used;
  unsigned char *h_imageInput, *d_imageInput, *d_imageGray;
  unsigned char *d_Gx, *d_Gy, *h_G, *d_G; // Sobel Operators
  //int *d_XMask, *d_YMask;
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

  /////////////////////////////////////////////////////////////////////////////////
  start = clock();

  h_imageInput = image.data;

  error = cudaMemcpy(d_imageInput, h_imageInput, sz, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    printf("Error copying data from h_imageInput to d_imageInput\n");
    exit(-1);
  }

  end = clock();
  time_used = ((double) (end - start)) /CLOCKS_PER_SEC;
  /////////////////////////////////////////////////////////////////////////////////



  //h_imageGray = (unsigned char*)malloc(size);

  error = cudaMalloc((void**)&d_imageGray, size);
  if (error != cudaSuccess) {
    printf("Error allocating memory for d_imageGray\n");
    exit(-1);
  }



  /////////////////////////////////////////////////////////////////////////////////
  start = clock();

  int blockSize = 32;
  dim3 dimBlock(blockSize, blockSize, 1);
  dim3 dimGrid(ceil(width/float(blockSize)), ceil(height/float(blockSize)), 1);
  img2grayCU<<<dimGrid,dimBlock>>>(d_imageInput, width, height, d_imageGray);
  cudaDeviceSynchronize();


  end = clock();
  time_used += ((double) (end - start)) /CLOCKS_PER_SEC;
  ///////////////////////////////////////////////////////////////////////////////////



/*
  error = cudaMemcpy(h_imageGray, d_imageGray, size, cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    printf("Error copying data from d_imageGray to h_imageGray\n");
    exit(-1);
  }
*/
  
  //-------------------- Masks -----------------------------

  char h_XMask[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
  char h_YMask[] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
//  char h_M[] = {1,0,-1,2,0,-2,1,0,-1};


//-------------------- copying to constant memory-------------------------
  start = clock();
  error = cudaMemcpyToSymbol(M, h_XMask, sizeof(char)*MASK_WIDTH*MASK_WIDTH);
  if(error != cudaSuccess){
      printf("Error copying mask h_M to M\n");
      exit(-1);
  }

  end = clock();
  time_used += ((double) (end - start)) /CLOCKS_PER_SEC;
  ///////////////////////////////////////////////////////////////////////////////////
/*
  error = cudaMemcpyToSymbol(YM, h_YMask, sizeof(char)*MASK_WIDTH*MASK_WIDTH);
  if(error != cudaSuccess){
      printf("Error copying mask h_YMask to M\n");
      exit(-1);
  }
*/


  //------------------------ Sobel --------------------------------

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

  h_G = (unsigned char*)malloc(size);

  error = cudaMalloc((void**)&d_G, size);
  if (error != cudaSuccess) {
    printf("Error allocating memory for d_G\n");
    exit(-1);
  }

  //-------------------------------------------------------------------


  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
  start = clock();
  // Convolution
  sobelSharedMem<<<dimGrid,dimBlock>>>(d_imageGray, width, height, MASK_WIDTH, d_Gx);
  cudaDeviceSynchronize();

  error = cudaMemcpyToSymbol(M, h_YMask, sizeof(char)*MASK_WIDTH*MASK_WIDTH);
  if(error != cudaSuccess){
      printf("Error copying mask h_YMask to M\n");
      exit(-1);
  }

  // Convolution in Gy
  sobelSharedMem<<<dimGrid,dimBlock>>>(d_imageGray, width, height, MASK_WIDTH, d_Gy);
  cudaDeviceSynchronize();

  // Union of Gx and Gy results
  UnionCU<<<dimGrid,dimBlock>>>(d_G, d_Gx, d_Gy, height, width);
  cudaDeviceSynchronize();


  error = cudaMemcpy(h_G, d_G, size, cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    printf("Error copying data from d_G to h_G\n");
    exit(-1);
  }

  end = clock();
  time_used += ((double) (end - start)) /CLOCKS_PER_SEC;
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

  //Mat result_Sobel;
  //result_Sobel.create(height, width, CV_8UC1);
  //result_Sobel.data = h_G;

  //imwrite("Sobel_Shared.png", result_Sobel);

  // printf("elapsed time: %lf", time_used);
  printf ("%lf \n",time_used);


  free(h_imageInput);
  cudaFree(d_imageInput);
  //free(h_imageGray);
  cudaFree(d_imageGray);
  //cudaFree(d_XMask);
  //cudaFree(d_YMask);
  free(h_G);
  cudaFree(d_Gx);
  cudaFree(d_Gy);
  cudaFree(d_G);

  return 0;
}

