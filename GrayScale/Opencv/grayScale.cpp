//#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
//#include <cv.h>
//#include <highgui.h>
//#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>

using namespace cv;

int main( int argc, char** argv )
{
 char* imageName = argv[1];

 Mat image;
 image = imread( imageName, 1 );

 if( argc != 2 || !image.data )
 {
   printf( " No image data \n " );
   return -1;
 }

 Mat gray_image;
 cvtColor( image, gray_image, CV_BGR2GRAY );

 imwrite( "./Gray_Image.jpg", gray_image );

 return 0;
}
