//Filtro Sobel GPU
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


__device__
unsigned char clamp(int value){
    if(value < 0)
        value = 0;
    else
        if(value > 255)
            value = 255;
    return (unsigned char)value;
}

__host__ 
void imprime(unsigned char *A,int filas, int columnas){//imprime los pixeles, 0..255
	for (int i = 0; i < filas; i++) {
            for (int j = 0; j < columnas; j++) {
                cout<<((int)A[(i * columnas) + j])<<" ";//le hacemos un cast para que me muestre los numeros y no los caracteres
            }
            cout<<endl;
        }
}


__global__
void convolucion(unsigned char *imagen, int *mascara, int filas, int columnas, unsigned char *resultado){
    
    int i = blockIdx.y*blockDim.y+threadIdx.y;
    int j = blockIdx.x*blockDim.x+threadIdx.x;
    int suma = 0;
    
    if(i < filas && j < columnas){
        		
        int aux_cols = j - 1, aux_rows = i - 1;
        for(int k = 0; k < 3; k++){//mask_rows
            for(int l = 0; l < 3; l++){//mask_cols
                if(aux_rows >= 0 && aux_cols >= 0 && aux_rows < filas && aux_cols < columnas)
                    suma += mascara[(k*3)+l]*imagen[(aux_rows*columnas)+ aux_cols];
                                        
                aux_cols++;
            }
            aux_rows++;
            aux_cols = j - 1;
        }
        resultado[(i * columnas) + j] = clamp(suma);
    }
}


__global__ 
void img2gray(unsigned char *imageInput, int width, int height, unsigned char *imageOutput){

    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    if((row < height) && (col < width)){
        
        imageOutput[row*width+col] = imageInput[(row*width+col)*3+RED]*0.299 + imageInput[(row*width+col)*3+GREEN]*0.587 
        + imageInput[(row*width+col)*3+BLUE]*0.114;
    }
}


__global__
void Union(unsigned char *img_resultado, unsigned char *resultado_Gx, unsigned char *resultado_Gy, int filas, int columnas){
        
    int i = blockIdx.y*blockDim.y+threadIdx.y;
    int j = blockIdx.x*blockDim.x+threadIdx.x;
    
    if(i < filas && j < columnas){
        img_resultado[(i * columnas) + j] = sqrtf((resultado_Gx[(i * columnas) + j] * resultado_Gx[(i * columnas) + j]) + (resultado_Gx[(i * columnas) + j] * resultado_Gx[(i * columnas) + j]) );
    }
}

int main(int argc, char **argv){
    
    cudaError_t error = cudaSuccess;//Para controlar errores
    unsigned char *h_ImagenInicial, *d_ImagenInicial;
    unsigned char *h_img_gray, *d_img_gray;//Imagen que vamos a pasar a escala de grises
    int *h_Mascara_X, *h_Mascara_Y,*d_Mascara_X, *d_Mascara_Y;
    char* imageName = argv[1];
    Mat image;//Imagen leída
    unsigned char *d_Gx, *d_Gy, *h_Gx, *h_Gy, *h_G, *d_G;//Operadores Sobel
    
    image = imread(imageName, 1);
    
    if(argc !=2 || !image.data){
        printf("No image Data \n");
        return -1;
    }
    
    //------------------imágenes--------------------------------
    
    //Sacamos los atributos de la imágen
    Size s = image.size(); 

    int width = s.width;
    int height = s.height;
    int sz = sizeof(unsigned char)*width*height*image.channels();
    int size = sizeof(unsigned char)*width*height;//para la imagen en escala de grises
    
    //Separamos memoria para la imagen inicial en el host y device
    h_ImagenInicial = (unsigned char*)malloc(sz);
    
    error = cudaMalloc((void**)&d_ImagenInicial,sz);
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_ImagenInicial\n");
        exit(-1);
    }
    
    //Pasamos los datos de la imágen leída 
    h_ImagenInicial = image.data;
    
    //Copiamos los datos al device
    error = cudaMemcpy(d_ImagenInicial,h_ImagenInicial,sz, cudaMemcpyHostToDevice);
    if(error != cudaSuccess){
        printf("Error copiando los datos de h_ImagenInicial a d_ImagenInicial \n");
        exit(-1);
    }
    
    //Separamos memoria para las imágenes a grises en el host y device
    h_img_gray = (unsigned char*)malloc(size);
    
    error = cudaMalloc((void**)&d_img_gray,size);
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_img_gray\n");
        exit(-1);
    }
    
    dim3 dimBlock(32,32,1);//bloque de 32 x 32 hilos = 1024 hilos
    dim3 dimGrid(ceil(width/float(32)),ceil(height/float(32)),1); 
    img2gray<<<dimGrid,dimBlock>>>(d_ImagenInicial, width, height, d_img_gray);
    cudaDeviceSynchronize();
    
    //Copiamos datos de la imágen a escala de grises del device al host
    error = cudaMemcpy(h_img_gray,d_img_gray,size, cudaMemcpyDeviceToHost);
    if(error != cudaSuccess){
        printf("Error copiando los datos de d_img_gray a h_img_gray \n");
        exit(-1);
    }
    
    /*
    //Mostramos la imagen en escala de grises
    Mat resultado_gray_image;
    resultado_gray_image.create(height,width,CV_8UC1);
    resultado_gray_image.data = h_img_gray;
    
    imshow("Grises",resultado_gray_image);
        
    waitKey(0);
    */
    
    //--------------------Máscaras-----------------------------
    
    //Separamos memoria para la máscara en X y Y en el host y device
    h_Mascara_X = (int*)malloc(3*3*sizeof(int));
    h_Mascara_Y = (int*)malloc(3*3*sizeof(int)); 
        
    error = cudaMalloc((void**)&d_Mascara_X,3*3*sizeof(int));
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_Mascara_X\n");
        exit(-1);
    }
    
    error = cudaMalloc((void**)&d_Mascara_Y,3*3*sizeof(int));
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_Mascara_Y\n");
        exit(-1);
    }
    
    
    //Inicializamos las máscaras
    h_Mascara_X[0]=-1;h_Mascara_X[1]=0;h_Mascara_X[2]=1;
    h_Mascara_X[3]=-2;h_Mascara_X[4]=0;h_Mascara_X[5]=2;
    h_Mascara_X[6]=-1;h_Mascara_X[7]=0;h_Mascara_X[8]=1;

    h_Mascara_Y[0]=-1;h_Mascara_Y[1]=-2;h_Mascara_Y[2]=-1;
    h_Mascara_Y[3]=0;h_Mascara_Y[4]=0;h_Mascara_Y[5]=0;
    h_Mascara_Y[6]=1;h_Mascara_Y[7]=2;h_Mascara_Y[8]=1;
    
    //Copiamos datos de las máscaras del host al device
    
    error = cudaMemcpy(d_Mascara_X,h_Mascara_X,3*3*sizeof(int), cudaMemcpyHostToDevice);
    if(error != cudaSuccess){
        printf("Error copiando los datos de h_Mascara_X a d_Mascara_X \n");
        exit(-1);
    }
    
    error = cudaMemcpy(d_Mascara_Y,h_Mascara_Y,3*3*sizeof(int), cudaMemcpyHostToDevice);
    if(error != cudaSuccess){
        printf("Error copiando los datos de h_Mascara_Y a d_Mascara_Y \n");
        exit(-1);
    }
    
    //-------------------------Sobel---------------------------------
    
    //Separamos memoria para G en el host y en el device
    h_G = (unsigned char*)malloc(size);
        
    error = cudaMalloc((void**)&d_G,size);
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_G\n");
        exit(-1);
    }
    
    //Separamos memoria para Gx y Gy en el host y device
    h_Gx = (unsigned char*)malloc(size); 
    h_Gy = (unsigned char*)malloc(size);
        
    error = cudaMalloc((void**)&d_Gx,size);
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_Gx\n");
        exit(-1);
    }
    
    error = cudaMalloc((void**)&d_Gy,size);
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_Gy\n");
        exit(-1);
    }
    
    //Lanzamos el Kernel para la convolucion en Gx
    //Con las mismas especificaciones del kernel que lanzamos para la imagen en escala de grises
    convolucion<<<dimGrid,dimBlock>>>(d_img_gray, d_Mascara_X, height, width, d_Gx);
    cudaDeviceSynchronize();
    
    //Lanzamos el Kernel para la convolucion en Gy
    convolucion<<<dimGrid,dimBlock>>>(d_img_gray, d_Mascara_Y, height, width, d_Gy);
    cudaDeviceSynchronize();
    
    //Unimos resultados 
    Union<<<dimGrid,dimBlock>>>(d_G,d_Gx,d_Gy,height,width);
    cudaDeviceSynchronize();
    
    //Copiamos datos de d_G a h_G
    error = cudaMemcpy(h_G, d_G, size,cudaMemcpyDeviceToHost);
    if(error != cudaSuccess){
        printf("Error copiando los datos de d_G a h_G \n");
        exit(-1);
    }
    
    /*
    //Mostramos el resultado del filtro Sobel
    Mat resultado_Sobel;
    resultado_Sobel.create(height,width,CV_8UC1);
    resultado_Sobel.data = h_G;
    
    imshow("Sobel",resultado_Sobel);
        
    waitKey(0);
    */
    
    //Liberamos memoria
    
    free(h_ImagenInicial); 
    cudaFree(d_ImagenInicial);
    free(h_img_gray); 
    cudaFree(d_img_gray);
    free(h_Mascara_X); 
    free(h_Mascara_Y); 
    cudaFree(d_Mascara_X); 
    cudaFree(d_Mascara_Y);
    free(h_Gx); 
    free(h_Gy); 
    free(h_G); 
    cudaFree(d_Gx); 
    cudaFree(d_Gy);
    cudaFree(d_G);
    
    return 0;
    
}