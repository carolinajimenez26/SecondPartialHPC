#include <opencv2/highgui.hpp>
#include <iostream>

int main( int argc, char** argv ) {

  //read the image
  cv::Mat image;
  image = cv::imread("Tux.png" , CV_LOAD_IMAGE_COLOR);

  //verify if the image was found
  if(! image.data ) {
      std::cout <<  "Could not open or find the image" << std::endl ;
      return -1;
    }

  //Name window
  cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );

  //Display image in the window created
  cv::imshow( "Display window", image );

  cv::waitKey(0);
  return 0;
}
