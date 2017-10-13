#include<iostream>
#include<stdio.h>
#include<malloc.h>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;



#define RED 2
#define GREEN 1
#define BLUE 0

#define MASK_WIDTH 3

//	__constant__ char M[MASK_WIDTH * MASK_WIDTH];

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
    imageOutput[(i * cols) + j] = sqrtf((Gx[(i * cols) + j] * Gx[(i * cols) + j]) + (Gx[(i * cols) + j] * Gx[(i * cols) + j]) );
  }
}


int main(int argc, char const *argv[])
{


  if (argc != 2) {
  	printf("Usage: Image path\n");
  	return 1;
	}

	///////////////////////declaracion de variables ////////////////////////////


	cudaError_t error = cudaSuccess;

	//times
	clock_t start, end;
  	double time_used;
  	char* imageName = argv[1];

  	//imagen inicial
  	unsigned char *h_ImageInit;
  	unsigned char *d_ImageInit;

  	//imagen en grises
  	unsigned char *d_imageGray;
  	//unsigned char *h_imageGray;

  	//imagenes con filtro en X y en Y
  	unsigned char *d_Gx, *d_Gy;

  	//imagen final
  	unsigned char *h_G, *d_G; 

  	//mascaras device
  	int *d_XMask, *d_YMask;

  	//mascaras device
  	int h_XMask[MASK_WIDTH*MASK_WIDTH] = {-1, 0, 1,-2, 0, 2,-1, 0, 1};
  	int h_YMask[MASK_WIDTH*MASK_WIDTH] = {-1,-2,-1, 0, 0, 0, 1, 2, 1};

  	//int h_XMask[3*3] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
  	//int h_YMask[3*3] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

  	char* imageName = argv[1];


  	//carga la imagen inicial
  	Mat image;
  	image = imread(imageName, 1);

	if (!image.data) {
		printf("No image Data\n");
	    return 1;
	}

	//se toman los parametros de la imagen
	Size s = image.size();
	int width = s.width;
	int height = s.height;

	int size = sizeof(unsigned char) * width * height * image.channels();
  	int sizeGray = sizeof(unsigned char) * width * height;


  	///////////reserve memory for Host and device ///////////////////////////


	//Imagen inicial en el Host
	h_ImageInit = (unsigned char*)malloc(size);
	//imagen final  host
	h_G = (unsigned char*)malloc(size);



	///////////////////// cudaMalloc ////////////////////////////////////////
	//imagen inicial device
	error = cudaMalloc((void**)&d_ImageInit,size);
  	if (error != cudaSuccess) {
    	printf("Error allocating memory for d_imageInput\n");
    	exit(-1);
  	}

  	//imagen en grises device
  	 error = cudaMalloc((void**)&d_imageGray, size);
  if (error != cudaSuccess) {
    printf("Error allocating memory for d_imageGray\n");
    exit(-1);
  }

  //Mascara en x 
  error = cudaMalloc((void**)&d_XMask, 3*3*sizeof(int));
  if (error != cudaSuccess) {
    printf("Error allocating memory for d_XMask\n");
    exit(-1);
  }

  //mascara en Y
  error = cudaMalloc((void**)&d_YMask, 3*3*sizeof(int));
  if (error != cudaSuccess) {
    printf("Error reservando memoria para d_Mascara_Y\n");
    exit(-1);
  }

  //convoluciones//

  //imagen convolucion Gx device
  error = cudaMalloc((void**)&d_Gx, size);
  if (error != cudaSuccess) {
    printf("Error allocating memory for d_Gx\n");
    exit(-1);
  }

  //imagen convolucion Gy device
  error = cudaMalloc((void**)&d_Gy, size);
  if (error != cudaSuccess) {
    printf("Error allocating memory for d_Gy\n");
    exit(-1);
  }


  //imagen final en device Union
  error = cudaMalloc((void**)&d_G, size);
  if (error != cudaSuccess) {
    printf("Error allocating memory for d_G\n");
    exit(-1);
  }

  /////////////////CudaMemCpy//////////////////////////////////////////////


  //carga la imagen inicial
  h_imageInput = image.data;

  error = cudaMemcpy(d_imageInput, h_imageInput, sz, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    printf("Error copiando  imagen inicial de host a device\n");
    exit(-1);
  }


 //mascaras
  error = cudaMemcpy(d_XMask, h_XMask, 3*3*sizeof(int), cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    printf("Error copiando mascara X  de host a device\n");
    exit(-1);
  }

  error = cudaMemcpy(d_YMask, h_YMask, 3*3*sizeof(int), cudaMemcpyHostToDevice);
  if(error != cudaSuccess){
    printf("Error copiando mascara Y  de host a device\n");
    exit(-1);
  }

  //////////////////////////////Grises//////////////////////////////////////

  int blockSize = 32;
  dim3 dimBlock(blockSize, blockSize, 1);
  dim3 dimGrid(ceil(width/float(blockSize)), ceil(height/float(blockSize)), 1);
  img2grayCU<<<dimGrid,dimBlock>>>(d_ImageInit, width, height, d_imageGray);
  cudaDeviceSynchronize();

  ////////////////////////////Convoluciones//////////////////////////////////

  // Convolution in Gx
  convolutionCU<<<dimGrid,dimBlock>>>(d_imageGray, d_XMask, height, width, d_Gx);
  cudaDeviceSynchronize();

  // Convolution in Gy
  convolutionCU<<<dimGrid,dimBlock>>>(d_imageGray, d_YMask, height, width, d_Gy);
  cudaDeviceSynchronize();


  // Union of Gx and Gy  ///// Sobel
  UnionCU<<<dimGrid,dimBlock>>>(d_G, d_Gx, d_Gy, height, width);
  cudaDeviceSynchronize();

  //Resultado de
  error = cudaMemcpy(h_G, d_G, size, cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    printf("Error copiando resultado  del device al host\n");
    exit(-1);
  }


  //crea la imagen resultante
  Mat result_Sobel;
  result_Sobel.create(height, width, CV_8UC1);
  result_Sobel.data = h_G;

  imwrite("Sobel_const.jpg", result_Sobel);

  //liberar memoria

  free(h_imageInput);
  //free(h_imageGray);
  free(h_G);


  cudaFree(d_imageInput);  
  cudaFree(d_imageGray);
  cudaFree(d_XMask);
  cudaFree(d_YMask);  
  cudaFree(d_Gx);
  cudaFree(d_Gy);
  cudaFree(d_G);


	return 0;
}