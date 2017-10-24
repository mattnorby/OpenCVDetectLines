#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <math.h>

int main(int argc, char *argv[]) {
    // Read the image and resize to 800x600 for easy viewing
    cv::Mat img = cv::imread("wheel.jpg", CV_LOAD_IMAGE_COLOR);
    if (img.empty()) return -1;
    cv::resize(img, img, cv::Size(800,600));

    // Find contours, using Canny.
    // Canny thresholds are higher than normal here because we are interested
    // in the lights, not the steel structure.
    // Canny output is a matrix (cv::Mat).
    cv::Mat cannyOutput;
    cv::Canny(img,   // input image
        cannyOutput, // output image
        250,         // lower threshold
        500);        // upper threshold

    // Search the Canny output for contours.
    // This operation returns a vector of contours (i.e. a vector of vector of line end points).
    // At this stage, each contour consists of multiple lines (approximating a curve).
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(cannyOutput, // input image (output from Canny)
        contours,                 // output: detected contours
        hierarchy,                // output: hierarchy of contours
        CV_RETR_TREE,             // mode constant
        CV_CHAIN_APPROX_SIMPLE,   // method constant
        cv::Point(0,0));          // x and y offset, as a point

    // Draw the detected contours on a new image.
    cv::Mat contourImg(600, // height of new image
        800,                // width of new image
        CV_8UC1,            // bits and channels for each pixel (here, 8-bit grayscale)
        cv::Scalar(0));     // background color (black)
    for (int i = 0; i < contours.size(); i++) {
      cv::drawContours(contourImg, // input image
          contours,              // array of contours (which are also arrays)
          i,                     // index of the contour to draw
          cv::Scalar(255),       // foreground color
          2);                    // width of drawn contour, in pixels
    }

    // Get rid of some extra thickness on the detected lines.
    // Doing this now reduces the number of duplicate lines we will get later.
    cv::erode(contourImg,   // input array
        contourImg,         // output array
        cv::Mat(),          // kernel
        cv::Point(-1, -1),  // anchor point (none)
        2);                 // number of iterations of erode

    // Detect lines, using the contour image we drew earlier.
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(contourImg,  // image to search
        lines,  // vector where the lines will be stored
        1,      // pixel resolution of the accumulator
        M_PI/180, // radians resolution of the accumulator (1 degree)
        80,     // minimum number of votes to consider a line
        100,    // minimum line length in pixels
        20);    // maximum gap along a single line, in pixels

    // Draw the detected lines.  To make it easier to see, we
    // draw on a black image rather than the original image.
    cv::Mat blackScreen(600, 800, CV_8UC3, cv::Scalar(0,0,0));
    std::vector<cv::Vec4i>::const_iterator iter = lines.begin();
    while (iter != lines.end()) {
      cv::Point pt1((*iter)[0],(*iter)[1]);
      cv::Point pt2((*iter)[2],(*iter)[3]);
      cv::line(blackScreen, pt1, pt2, cv::Scalar(0,255,0));
      ++iter;
    }

    // Calculate the intersection point of all those lines, using least squares.
    // To use the OpenCV solve routine, we need to make an n x 2 matrix A and
    // a 2 x 1 vector b such that Av = b, where v = the 2 x 1 answer [x y].
    cv::Mat A(lines.size(), 2, CV_32FC1);
    cv::Mat b(lines.size(), 1, CV_32FC1);
    int n = 0;
    for (iter = lines.begin(); iter != lines.end(); iter++) {
      cv::Point pt1((*iter)[0],(*iter)[1]);
      cv::Point pt2((*iter)[2],(*iter)[3]);
      if (pt1.x == pt2.x) {
        // Vertical line --> equation is x = constant --> 1x + 0y = b
        A.at<float>(n, 0) = 1.0f;
        A.at<float>(n, 1) = 0.0f;
        b.at<float>(n, 0) = pt1.x;
      } else {
        // Non-vertical line.  Calculate slope to get y = mx + b --> -mx + y = b
        // Coefficients in each row of A will be -m and 1.
        float m = ((float) (pt1.y - pt2.y)) / (pt1.x - pt2.x);
        A.at<float>(n, 0) = -m;
        A.at<float>(n, 1) = 1.0f;
        b.at<float>(n, 0) = pt1.y - m * pt1.x;
      }
      n++;
    }

    // Find the point that is a minimum distance from all of the lines.
    cv::Mat v;
    if (!cv::solve(A,   // matrix on the LHS of Av = b
        b,              // vector on the RHS of Av = b
        v,       // vector we are solving for (v)
        cv::DECOMP_SVD) // decomposition method for matrix A
        ) {
      std::cout << "solving failed" << std::endl;
      return -1;
    }

    // Print out the solution x and y coordinates
    cv::Point solutionPt(v.at<float>(0,0), v.at<float>(1,0));
    std::cout << "solution = " << solutionPt.x << ", " << solutionPt.y << std::endl;

    // Show the detected lines and center
    cv::circle(blackScreen, solutionPt, 5, cv::Scalar(0,255,255));
    cv::namedWindow("Detect Lines");
    cv::imshow("Detect Lines", blackScreen);

    // Show the contour image
    cv::namedWindow("Contours");
    cv::imshow("Contours", contourImg);

    // Show the original image with detected center
    cv::circle(img, solutionPt, 5, cv::Scalar(0,255,255));
    cv::namedWindow("Original");
    cv::imshow("Original", img);

    cv::waitKey(0);
    return 0;
}
