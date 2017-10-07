//#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std
using namespace cv


int main( int argc, char** argv ) {

  //read the image
  Mat image;
  image = imread("Tux.png" , CV_LOAD_IMAGE_COLOR);

  //verify if the image was found
  if(! image.data ) {
      cout <<  "Could not open or find the image" << std::endl ;
      return -1;
    }
  
  imwrite("./ImageOpenCv.jpg",image);
  

  //Name window
  //cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );

  //Display image in the window created
  //cv::imshow( "Display window", image );

  //cv::waitKey(0);
  return 0;
}
