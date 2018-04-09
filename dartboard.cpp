/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - face.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

float circle3D[1280][1280][256];
float circle2D[1280][1280];
float orientation[1280][1280];
int circleX[16];
int circleY[16];
int circleR[16];
int lineX[1280];
int lineY[1280];

/** Function Headers */
void detectAndDisplay( Mat frame );

void GaussianBlur(cv::Mat &input,
                  int size,
                  cv::Mat &blurredOutput);

void Hough(cv::Mat &magnitude,
           int maxradius,
           int minradius,
           cv::Mat &houghspace);

void detectCircle(cv::Mat &hough,
                  cv::Mat &original,
                  int threshold,
                  int maxradius,
                  int minradius);

void detectline(cv::Mat &source,
                cv::Mat &output);

void findOrientation(cv::Mat &direction,
                     cv::Mat &blur_image);


/** Global variables */
String cascade_name = "cascade.xml";
CascadeClassifier cascade;


/** @function main */
int main( int argc, const char** argv )
{
    // 1. Read Input Image
    Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat frame2 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    
    Mat gray_image;
    cvtColor( frame, gray_image, CV_BGR2GRAY );
    
    Mat blur_image, magnitude, direction, houghspace, detectedimage, linesDetected;
    GaussianBlur(gray_image, 5, blur_image);
    GaussianBlur(gray_image, 5, magnitude);
    GaussianBlur(gray_image, 5, direction);
    
    findOrientation(direction, blur_image);
    Canny(frame, magnitude, 200, 600, 3);
    Hough(magnitude, 90, 20, houghspace);
    
    detectCircle(houghspace, frame, 45, 90, 20);
    detectline(frame, linesDetected);
    
    // 2. Load the Strong Classifier in a structure called `Cascade'
    if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
    
    // 3. Detect Faces and Display Result
    detectAndDisplay( frame );
    
    // 4. Save Result Image
    imwrite( "magnitude.jpg", magnitude );
    imwrite( "orientation.jpg", direction);
    imwrite( "houghspace.jpg", houghspace );
    imwrite( "detected.jpg", frame );
    imwrite( "linesDetected.jpg", linesDetected);
    
    return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame )
{
    std::vector<Rect> faces;
    Mat frame_gray;
    
    // 1. Prepare Image by turning it into Grayscale and normalising lighting
    cvtColor( frame, frame_gray, CV_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    
    // 2. Perform Viola-Jones Object Detection
    cascade.detectMultiScale( frame_gray, faces, 1.01, 6, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );
    
    // 3. Print number of Faces found
    printf("%lu\n", faces.size());
    
    int j=0;
    int shortestDistance=1000;
    int l;
    int BmidX;
    int BmidY;
    
    // combine circle hough to haarlike
    while (circleX[j]!=0) {
        for( int i = 0; i < faces.size(); i++ )
        {
            int midY = (faces[i].x + faces[i].x + faces[i].width)/2;
            int midX = (faces[i].y + faces[i].y + faces[i].height)/2;
            
            // find the distance between the centre point in hough space and the centre point of haarlike detection box
            int distance = sqrt((circleX[j]-midX)*(circleX[j]-midX)+(circleY[j]-midY)*(circleY[j]-midY));
            if (distance<shortestDistance)
            {
                shortestDistance=distance;
                l=i;
                BmidX=midX;
                BmidY=midY;
            }
        }
        
        // if the centre point in hough space is in the box
        if (shortestDistance<faces[l].width) {
            
            //draw box
            rectangle(frame, Point(circleY[j]-circleR[j], circleX[j]+circleR[j]), Point(circleY[j]+circleR[j], circleX[j]-circleR[j]), Scalar( 0, 255, 0 ), 2);
            shortestDistance=1000;
            
            // erase lines in the detected circles
            for (int i=0; i<1280; i++) {
                
                if (lineY[i] > circleY[j]-circleR[j]-50 && lineY[i] < circleY[j]+circleR[j]+50) {
                    if (lineX[i] > circleX[j]-circleR[j]-50 && lineX[i] < circleX[j]+circleR[j]+50) {
                        
                        lineX[i]=0;
                        lineY[i]=0;
                    }
                }
            }
        }
        j++;
    }
    j=0;
    
    // combine line hough to haarlike
    int m;
    int Baverage=0;
    int average=0;
    
    for( int i = 0; i < faces.size(); i++ )
    {
        int midY = (faces[i].x + faces[i].x + faces[i].width)/2;
        int midX = (faces[i].y + faces[i].y + faces[i].height)/2;
        int j=0;
        int n=0;
        
        // find lines in the haarlike detection box
        while (j<1280) {
            if (sqrt((lineX[j]-midX)*(lineX[j]-midX)+(lineY[j]-midY)*(lineY[j]-midY))<50) {
                n++;
            }
            j++;
        }
        if (n>=20) {
            l=i;
            
            // calculate the average value of pixels in the hough space
            for (int i=faces[l].x; i<faces[l].x + faces[l].width; i++) {
                for (int j=faces[l].y; j<faces[l].y + faces[l].height; j++) {
                    average = average + circle2D[j][i];
                }
            }
            average = average/(faces[l].width*faces[l].height);
        }
        if (average>Baverage) {
            Baverage=average;
            m=l;
        }
        average=0;
    }
    
    // if the value of pixels in the detection box is large enough
    if (Baverage>8) {
        rectangle(frame, Point(faces[m].x, faces[m].y), Point(faces[m].x + faces[m].width, faces[m].y +faces[m].height), Scalar( 0, 0, 255 ), 2);
    }
}


void findOrientation(cv::Mat &direction, cv::Mat &blur_image)
{
    
    for(int i=0; i<direction.rows; i++) {
        for(int j=0; j<direction.cols; j++) {
            
            float xdirection = (blur_image.at<uchar>(i-1,j-1)*-1 +
                                blur_image.at<uchar>(i+1,j-1)*1 +
                                blur_image.at<uchar>(i-1,j)*-1 +
                                blur_image.at<uchar>(i+1,j)*1 +
                                blur_image.at<uchar>(i-1,j+1)*-1 +
                                blur_image.at<uchar>(i+1,j+1)*1);
            
            float ydirection = (blur_image.at<uchar>(i-1,j-1)*-1 +
                                blur_image.at<uchar>(i,j-1)*-1 +
                                blur_image.at<uchar>(i+1,j-1)*-1 +
                                blur_image.at<uchar>(i-1,j+1)*1 +
                                blur_image.at<uchar>(i,j+1)*1 +
                                blur_image.at<uchar>(i+1,j+1)*1);
            
            float pixel = atan(ydirection/xdirection);
            
            orientation[i][j]=pixel;
            
            direction.at<uchar>(i,j) = pixel*180/3.14;
        }
    }
}


void Hough(cv::Mat &magnitude, int maxradius, int minradius, cv::Mat &houghspace)
{
    
    houghspace.create(magnitude.size(), magnitude.type());
    
    // Find centres for every circle in hough space
    for(int i=0; i<magnitude.rows; i++) {
        for(int j=0; j<magnitude.cols; j++) {
            
            if (magnitude.at<uchar>(i,j)==255) {
                float theta = orientation[i][j];
                theta = theta*180/3.14;
                
                // Draw arcs in positive direction
                for (int r=minradius; r<maxradius; r++) {
                    for (int t=theta-5; t<theta+5; t++) {
                        int x = i + r*cos(t*3.14/180);
                        int y = j + r*sin(t*3.14/180);
                        if (x>0 & x<1280) {
                            if (y>0 & y<1280) {
                                circle3D[x][y][r] = circle3D[x][y][r] + 0.1;
                            }
                        }
                    }
                }
                // Draw arcs in negative direction
                for (int r=minradius; r<maxradius; r++) {
                    for (int t=theta-5; t<theta+5; t++) {
                        int x = i - r*cos(t*3.14/180);
                        int y = j - r*sin(t*3.14/180);
                        if (x>0 & x<1280) {
                            if (y>0 & y<1280) {
                                circle3D[x][y][r] = circle3D[x][y][r] + 0.1;
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Transform the 3D hough space to a 2D hough space
    for (int r=minradius; r<maxradius; r++) {
        for(int i=0; i<1280; i++) {
            for(int j=0; j<1280; j++) {
                circle2D[i][j] = circle2D[i][j]+circle3D[i][j][r];
            }
        }
    }
    
    // Set a limitation for the pixels and store the pixels as hough space image
    for(int i=0; i<magnitude.rows; i++) {
        for(int j=0; j<magnitude.cols; j++) {
            
            if (circle2D[i][j]>255) {
                circle2D[i][j]=255;
            }
            
            uchar hpixel = circle2D[i][j];
            houghspace.at<uchar>(i,j) = hpixel;
        }
    }
}


void detectCircle(cv::Mat &hough, cv::Mat &original, int threshold, int maxradius, int minradius)
{
    
    int max=0, n=0, x, y;
    float pixel=0, rad=0;
    
    // Repeat until the pixel of found centre is lower than the preset threshold
    while (1) {
        // Find the brightest pixel
        for(int i=0; i<original.rows; i++) {
            for(int j=0; j<original.cols; j++) {
                
                if (circle2D[i][j] >max) {
                    max = circle2D[i][j];
                    x = i;
                    y = j;
                }
            }
        }
        
        // Check if the value of centre is lower than threshold
        if (max<threshold) {
            break;
        }
        
        circleX[n]=x;
        circleY[n]=y;
        
        // Find the radius of the circle
        for (int r=minradius; r<maxradius; r++) {
            if (circle3D[x][y][r]>pixel) {
                pixel=circle3D[x][y][r];
                rad=r;
            }
        }
        
        circleR[n]=rad;
        n++;
        
        // Erase the pixels in the found circle in hough space
        for(int r=0; r<rad; r++) {
            for (int t=0; t<360; t++) {
                int i = x + r*cos(t*3.14/180);
                int j = y + r*sin(t*3.14/180);
                if (i>0 & i<original.rows) {
                    if (j>0 & j<original.cols) {
                        circle2D[i][j] = 0;
                    }
                }
            }
        }
        for(int r=0; r<rad; r++) {
            for (int t=0; t<360; t++) {
                int i = x+1 + r*cos(t*3.14/180);
                int j = y+1 + r*sin(t*3.14/180);
                if (i>0 & i<original.rows) {
                    if (j>0 & j<original.cols) {
                        circle2D[i][j] = 0;
                    }
                }
            }
        }
        
        // Reset the variables and repeat the loop
        max=0;
        x=0;
        y=0;
        pixel=0;
        rad=0;
    }
    printf("%d\n", n);
}


void detectline(cv::Mat &source, cv::Mat &cdst)
{
    Mat dst;
    Canny(source, dst, 200, 600, 3);
    cvtColor(dst, cdst, CV_GRAY2BGR);
    
    vector<Vec4i> lines;
    HoughLinesP(dst, lines, 15, 10*CV_PI/180, 80, 30, 8 );
    for( size_t i = 0; i < lines.size(); i++ )
    {
        Vec4i l = lines[i];
        line( cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 1, CV_AA);
        int lX=(l[0]+l[2])/2;
        int lY=(l[1]+l[3])/2;
        lineX[i]=lY;
        lineY[i]=lX;
    }
}


void GaussianBlur(cv::Mat &input, int size, cv::Mat &blurredOutput)
{
    // intialise the output using the input
    blurredOutput.create(input.size(), input.type());
    
    // create the Gaussian kernel in 1D
    cv::Mat kX = cv::getGaussianKernel(size, -1);
    cv::Mat kY = cv::getGaussianKernel(size, -1);
    
    // make it 2D multiply one by the transpose of the other
    cv::Mat kernel = kX * kY.t();
    
    //CREATING A DIFFERENT IMAGE kernel WILL BE NEEDED
    //TO PERFORM OPERATIONS OTHER THAN GUASSIAN BLUR!!!
    
    // we need to create a padded version of the input
    // or there will be border effects
    int kernelRadiusX = ( kernel.size[0] - 1 ) / 2;
    int kernelRadiusY = ( kernel.size[1] - 1 ) / 2;
    
    cv::Mat paddedInput;
    cv::copyMakeBorder( input, paddedInput,
                       kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
                       cv::BORDER_REPLICATE );
    
    // now we can do the convoltion
    for ( int i = 0; i < input.rows; i++ )
    {
        for( int j = 0; j < input.cols; j++ )
        {
            double sum = 0.0;
            for( int m = -kernelRadiusX; m <= kernelRadiusX; m++ )
            {
                for( int n = -kernelRadiusY; n <= kernelRadiusY; n++ )
                {
                    // find the correct indices we are using
                    int imagex = i + m + kernelRadiusX;
                    int imagey = j + n + kernelRadiusY;
                    int kernelx = m + kernelRadiusX;
                    int kernely = n + kernelRadiusY;
                    
                    // get the values from the padded image and the kernel
                    int imageval = ( int ) paddedInput.at<uchar>( imagex, imagey );
                    double kernalval = kernel.at<double>( kernelx, kernely );
                    
                    // do the multiplication
                    sum += imageval * kernalval;
                }
            }
            // set the output value as the sum of the convolution
            blurredOutput.at<uchar>(i, j) = (uchar) sum;
        }
    }
}


///////////
//14
//5
//9
//
//
///////////



