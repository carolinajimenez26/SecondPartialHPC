#include<iostream>
#include<stdio.h>
#include<malloc.h>
#include <cv.h>
#include <highgui.h>
#include<opencv2/opencv.hpp>
using namespace std;
using namespace cv;

unsigned char clamp(int value){//porque cuando se hace la convolución pueden salir números fuera del rango de unsigned char
    if(value < 0)
        value = 0;
    else
        if(value > 255)
            value = 255;
    return (unsigned char)value;
}

void imprime(unsigned char *A,int filas, int columnas){//imprime los pixeles, 0..255
	for (int i = 0; i < filas; i++) {
            for (int j = 0; j < columnas; j++) {
                cout<<((int)A[(i * columnas) + j])<<" ";//le hacemos un cast para que me muestre los numeros y no los caracteres
            }
            cout<<endl;
        }
}

void convolucion(unsigned char *imagen, int mascara[3][3], int filas, int columnas, unsigned char *resultado){

	for(int i = 0; i < filas; i++){

		for(int j = 0; j < columnas; j++){//hacemos el recorrido por cada pixel
                    int suma = 0;
                    int aux_cols = j - 1, aux_rows = i - 1;
                    for(int k = 0; k < 3; k++){//mask_rows
                            for(int l = 0; l < 3; l++){//mask_cols
                                    if((aux_rows >= 0 && aux_cols >= 0) && (aux_rows < filas && aux_cols < columnas))

                                        suma += mascara[k][l]*imagen[(aux_rows*columnas)+ aux_cols];

                                    aux_cols++;
                            }
                            aux_rows++;
                            aux_cols = j - 1;
                    }

                    resultado[(i * columnas) + j] = clamp(suma);
                }
	}
}

void Union(unsigned char *img_resultado, unsigned char *resultado_Gx, unsigned char *resultado_Gy, int filas, int columnas){
        for(int i = 0; i < filas; i++){
            for(int j = 0; j < columnas; j++){
                img_resultado[(i * columnas) + j] = sqrt(pow(resultado_Gx[(i * columnas) + j],2) + pow(resultado_Gx[(i * columnas) + j],2));
            }
        }
}

int main(int argc, char **argv){
	unsigned char *img_gray;
	unsigned char *G, *resultado_Gx , *resultado_Gy;//imagenes para la convolucion
	int Mascara_X[3][3], Mascara_Y[3][3];
	char* imageName = argv[1];
	Mat image;

  	image = imread(imageName, 1);

        if(argc !=2 || !image.data){
            printf("No image Data \n");
            return -1;
        }
        
        Size s = image.size();//sacamos los atributos de la imagen

        int width = s.width;
        int height = s.height;
        int size = sizeof(unsigned char)*width*height;//para la imagen en escala de grises

        //La pasamos a escala de grises con Opencv
        Mat gray_image_opencv;//Esta es la que le vamos a aplicar el filtro Sobel
        gray_image_opencv.create(height,width,CV_8UC1);
        cvtColor(image, gray_image_opencv, CV_BGR2GRAY);//pasamos la imagen que se lee a escala de grises

        img_gray = gray_image_opencv.data;//queda con el mismo height y weight de la imagen normal

        //imshow("Escala de grises",gray_image_opencv);

        resultado_Gx = (unsigned char*)malloc(size);
        resultado_Gy = (unsigned char*)malloc(size);
        G = (unsigned char*)malloc(size);

        Mascara_X[0][0]=-1;Mascara_X[0][1]=0;Mascara_X[0][2]=1;
        Mascara_X[1][0]=-2;Mascara_X[1][1]=0;Mascara_X[1][2]=2;
        Mascara_X[2][0]=-1;Mascara_X[2][1]=0;Mascara_X[2][2]=1;

        Mascara_Y[0][0]=-1;Mascara_Y[0][1]=-2;Mascara_Y[0][2]=-1;
        Mascara_Y[1][0]=0;Mascara_Y[1][1]=0;Mascara_Y[1][2]=0;
        Mascara_Y[2][0]=1;Mascara_Y[2][1]=2;Mascara_Y[2][2]=1;


        convolucion(img_gray, Mascara_X, height, width, resultado_Gx);
        convolucion(img_gray, Mascara_Y, height, width, resultado_Gy);
        Union(G,resultado_Gx,resultado_Gy,height,width);

        Mat resultado;
        resultado.create(height,width,CV_8UC1);

        resultado.data = G;
        imshow("Sobel",resultado);

        waitKey(0);

        //Se libera memoria
        free(G);free(resultado_Gx);free(resultado_Gy);

        return 0;
}
