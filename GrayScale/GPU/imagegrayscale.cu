#include <stdlib.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <time.h>
#include <cuda.h>

#define RED 2
#define GREEN 1
#define BLUE 0

#define Channels 3

using namespace cv;


__global__ void img2gray(unsigned char *imageInput, int width, int height, unsigned char *imageOutput){
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    if((row < height) && (col < width)){
	int pos = (row*width+col)*Channels;

        imageOutput[row*width+col] = imageInput[pos+RED]*0.299 + imageInput[pos+GREEN]*0.587 + imageInput[pos+BLUE]*0.114;
    }
}


int main(int argc, char **argv){


  cudaError_t error = cudaSuccess;

  //times
  clock_t start, end, startGPU, endGPU;
  double cpu_time_used, gpu_time_used;
  char* imageName = argv[1];


  unsigned char *h_ImageInit;//*dataRawImage
  unsigned char *d_ImageInit;//d_dataRawImage

  unsigned char *d_imageOutput;
  unsigned char *h_imageOutput;

  //se carga la imagen
  Mat image;
  image = imread(imageName, 1);

    if(argc !=2 || !image.data){
        printf("No image Data \n");
        return -1;
    }

  //se toman los parametros de la imagen
  Size s = image.size();
  int width = s.width;
  int height = s.height;

  //tamaño de la  imagen
  int size = sizeof(unsigned char)*width*height*image.channels(); //para la imagen normal (3 canales)
  int sizeGray = sizeof(unsigned char)*width*height;//para la imagen en escala de grises (1 canal)

  //reserve memory for Host and device ////////////////////////////////////////

  //Imagen inicial en el Host
  h_ImageInit = (unsigned char*)malloc(size);
  //Imagen de salida del Device
  h_imageOutput = (unsigned char *)malloc(sizeGray);

  //Imagen inicial en el device
  error = cudaMalloc((void**)&d_ImageInit,size);
  if(error != cudaSuccess){
      printf("Error reservando memoria para Imagen inicial en el device\n");
      exit(-1);
  }

    //Imagen salida en el device
  error = cudaMalloc((void**)&d_imageOutput,sizeGray);
  if(error != cudaSuccess){
      printf("Error reservando memoria para d_imageOutput\n");
      exit(-1);
  }
///////////////////////////////////////////////////////////////////////////////

  //se carga la imagen
  h_ImageInit = image.data;


  ////////////////////////Algoritmo Paralelo /////////////////////////////////

  //tiempo GPU
  startGPU = clock();

  //se copian los datos de la imagen del host al device
  error = cudaMemcpy(d_ImageInit, h_ImageInit,size, cudaMemcpyHostToDevice);
  if(error != cudaSuccess){
      printf("Error copiando los datos de dataRawImage a d_dataRawImage \n");
      exit(-1);
  }

    int blockSize = 32;
    dim3 dimBlock(blockSize,blockSize,1);////bloque de 32 x 32 hilos = 1024 hilos
    dim3 dimGrid(ceil(width/float(blockSize)),ceil(height/float(blockSize)),1);
    img2gray<<<dimGrid,dimBlock>>>(d_ImageInit,width,height,d_imageOutput);

    cudaDeviceSynchronize();
    //copian los datos de la imagen del device a la de salida del host
    cudaMemcpy(h_imageOutput,d_imageOutput,sizeGray,cudaMemcpyDeviceToHost);
    endGPU = clock();

    Mat gray_image;
    gray_image.create(height,width,CV_8UC1);
    gray_image.data = h_imageOutput;
    ////////////////////////Algoritmo Paralelo /////////////////////////////////


    ////////////////////////Algoritmo OpenCV /////////////////////////////////
    start = clock();
    Mat gray_image_opencv;
    cvtColor(image, gray_image_opencv, CV_BGR2GRAY);
    end = clock();
    ////////////////////////Algoritmo OpenCV /////////////////////////////////

    imwrite("./Gray_Image.jpg",gray_image);

   //display times
    gpu_time_used = ((double) (endGPU - startGPU)) / CLOCKS_PER_SEC;
    printf("Tiempo Algoritmo Paralelo: %.10f\n",gpu_time_used);
    cpu_time_used = ((double) (end - start)) /CLOCKS_PER_SEC;
    printf("Tiempo Algoritmo OpenCV: %.10f\n",cpu_time_used);
    printf("La aceleración obtenida es de %.10fX\n",cpu_time_used/gpu_time_used);

    free(h_ImageInit);
    free(h_imageOutput);
    cudaFree(d_ImageInit);
    cudaFree(d_imageOutput);
    return 0;
}
