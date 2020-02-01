/*
 Copyright 2017 BIG VISION LLC ALL RIGHTS RESERVED

 This program is distributed WITHOUT ANY WARRANTY to the
 students of the online course titled

 "Computer Visionfor Faces" by Satya Mallick

 for personal non-commercial use.

 Sharing this code is strictly prohibited without written
 permission from Big Vision LLC.

 For licensing and other inquiries, please email
 spmallick@bigvisionllc.com

 */

 #include <iostream>
 #include <fstream>
 #include <sstream>
 #include <math.h>
 #include <map>
 #include <iomanip> // setprecision
 #include <sstream> // stringstream

 #include <opencv2/core.hpp>
 #include <opencv2/videoio.hpp>
 #include <opencv2/highgui.hpp>
 #include <opencv2/imgproc.hpp>
 #include <opencv2/dnn.hpp>

 #include <dlib/string.h>
 #include <dlib/dnn.h>
 #include <dlib/image_io.h>
 #include <dlib/opencv.h>
 #include <dlib/image_processing.h>
 #include <dlib/image_processing/frontal_face_detector.h>
 #include "faceBlendCommon.hpp"

// dirent.h is pre-included with *nix like systems
// but not for Windows. So we are trying to include
// this header files based on Operating System
#ifdef _WIN32
  #include "dirent.h"
#elif __APPLE__
  #include "TargetConditionals.h"
#if TARGET_OS_MAC
  #include <dirent.h>
#else
  #error "Not Mac. Find an alternative to dirent"
#endif
#elif __linux__
  #include <dirent.h>
#elif __unix__ // all unices not caught above
  #include <dirent.h>
#else
  #error "Unknown compiler"
#endif

using namespace cv;
using namespace dlib;
using namespace std;

#define SKIP_FRAMES 1
#define recThreshold 0.8


// function to print a vector
template<typename T>
void printVector(std::vector<T>& vec) {
  for (int i = 0; i < vec.size(); i++) {
    cout << i << " " << vec[i] << "; ";
  }
  cout << endl;
}

// read names and labels mapping from file
static void readLabelNameMap(const string& filename, std::vector<string>& names, std::vector<int>& labels,
                             std::map<int, string>& labelNameMap, char separator = ';') {
  std::ifstream file(filename.c_str(), ifstream::in);
  if (!file) {
    string error_message = "No valid input file was given, please check the given filename.";
    CV_Error(CV_StsBadArg, error_message);
  }
  string line;
  string name, labelStr;
  // read lines from file one by one
  while (getline(file, line)) {
    stringstream liness(line);
    // read first word which is person name
    getline(liness, name, separator);
    // read second word which is integer label
    getline(liness, labelStr);
    if(!name.empty() && !labelStr.empty()) {
      names.push_back(name);
      // convert label from string format to integer
      int label = atoi(labelStr.c_str());
      labels.push_back(label);
      // add (integer label, person name) pair to map
      labelNameMap[label] = name;
    }
  }
}

// read descriptors saved on disk
static void readDescriptors(const string& filename, std::vector<int>& faceLabels, std::vector<Mat>& faceDescriptors, char separator = ';') {
  std::ifstream file(filename.c_str(), ifstream::in);
  if (!file) {
    string error_message = "No valid input file was given, please check the given filename.";
    CV_Error(CV_StsBadArg, error_message);
  }
  // each line has:
  // 1st element = face label
  // rest 128 elements = descriptor elements
  string line;
  string faceLabel;
  // valueStr = one element of descriptor in string format
  // value = one element of descriptor in float
  string valueStr;
  float value;
  std::vector<float> faceDescriptorVec;
  // read lines from file one by one
  while (getline(file, line)) {
    stringstream liness(line);
    // read face label
    // read first word on a line till separator
    getline(liness, faceLabel, separator);
    if(!faceLabel.empty()) {
      faceLabels.push_back(std::atoi(faceLabel.c_str()));
    }

    faceDescriptorVec.clear();
    // read rest of the words one by one using separator
    while (getline(liness, valueStr, separator)) {
      if (!valueStr.empty()) {
        // convert descriptor element from string to float
        faceDescriptorVec.push_back(atof(valueStr.c_str()));
      }
    }

    // convert face descriptor from vector of float to Dlib's matrix format
    Mat faceDescriptor(faceDescriptorVec);
    faceDescriptors.push_back(faceDescriptor.clone());
  }
}

// find nearest face descriptor from vector of enrolled faceDescriptor
// to a query face descriptor
void nearestNeighbor(Mat& faceDescriptorQuery,
                    std::vector<Mat>& faceDescriptors,
                    std::vector<int>& faceLabels, int& label, float& minDistance) {
  int minDistIndex = 0;
  minDistance = 1.0;
  label = -1;
  // Calculate Euclidean distances between face descriptor calculated on face dectected
  // in current frame with all the face descriptors we calculated while enrolling faces
  // Calculate minimum distance and index of this face
  for (int i = 0; i < faceDescriptors.size(); i++) {
    double distance = cv::norm(faceDescriptors[i].t() - faceDescriptorQuery);
    if (distance < minDistance) {
      minDistance = distance;
      minDistIndex = i;
    }
  }
  // if minimum distance is greater than a threshold
  // assign integer label -1 i.e. unknown face
  if (minDistance > recThreshold){
    label = -1;
  } else {
    label = faceLabels[minDistIndex];
  }
}

int main(int argc, const char** argv) {

  // Initialize face detector, facial landmarks detector and face recognizer
  const std::string recModelPath = "../data/models/openface.nn4.small2.v1.t7";
  frontal_face_detector faceDetector = get_frontal_face_detector();
  dnn::Net recModel = dnn::readNetFromTorch(recModelPath);
  dlib::shape_predictor landmarkDetector;
  dlib::deserialize("../data/models/shape_predictor_5_face_landmarks.dat") >> landmarkDetector;


  // read names, labels and labels-name-mapping from file
  std::map<int, string> labelNameMap;
  std::vector<string> names;
  std::vector<int> labels;
  const string labelNameFile = "label_name_openface.txt";
  readLabelNameMap(labelNameFile, names, labels, labelNameMap);

  // read descriptors of enrolled faces from file
  const string faceDescriptorFile = "descriptors_openface.csv";
  std::vector<int> faceLabels;
  std::vector<Mat> faceDescriptors;
  readDescriptors(faceDescriptorFile, faceLabels, faceDescriptors);
  printVector(faceDescriptors);
  // Create a VideoCapture object

  VideoCapture cap;
  cap.open("../data/videos/face1.mp4");

  // Check if OpenCV is able to read feed from camera
  if (!cap.isOpened()) {
    cerr << "Unable to connect to camera" << endl;
    return 1;
  }

  int count = 0;
  double t = cv::getTickCount();

  while (1) {
    t = cv::getTickCount();
    // Capture frame
    Mat im;
    cap >> im;

    // If the frame is empty, break immediately
    if (im.empty()){
      break;
    }

    // We will be processing frames after an interval
    // of SKIP_FRAMES to increase processing speed
    if ((count % SKIP_FRAMES) == 0) {

      cv_image<bgr_pixel> imDlib(im);

      // detect faces in image
      std::vector<dlib::rectangle> faceRects = faceDetector(imDlib);
      // Now process each face we found
      for (int i = 0; i < faceRects.size(); i++) {
        cout << faceRects.size() << " Face(s) Found" << endl;

        Mat alignedFace;
        alignFace(im, alignedFace, faceRects[i], landmarkDetector, cv::Size(96, 96));
        cv::Mat blob = dnn::blobFromImage(alignedFace, 1.0/255, cv::Size(96, 96), Scalar(0,0,0), false, false);
        recModel.setInput(blob);
        Mat faceDescriptorQuery = recModel.forward();

        // Find closest face enrolled to face found in frame
        int label;
        float minDistance;
        nearestNeighbor(faceDescriptorQuery, faceDescriptors, faceLabels, label, minDistance);
        // Name of recognized person from map
        string name = labelNameMap[label];

        cout << "Time taken = " << ((double)cv::getTickCount() - t)/cv::getTickFrequency() << endl;

        // Draw a rectangle for detected face
        Point2d p1 = Point2d(faceRects[i].left(), faceRects[i].top());
        Point2d p2 = Point2d(faceRects[i].right(), faceRects[i].bottom());
        cv::rectangle(im, p1, p2, Scalar(0, 0, 255), 1, LINE_8);

        // Draw circle for face recognition
        Point2d center = Point((faceRects[i].left() + faceRects[i].right())/2.0,
        (faceRects[i].top() + faceRects[i].bottom())/2.0 );
        int radius = static_cast<int> ((faceRects[i].bottom() - faceRects[i].top())/2.0);
        cv::circle(im, center, radius, Scalar(0, 255, 0), 1, LINE_8);

        // Write text on image specifying identified person and minimum distance
        stringstream stream;
        stream << name << " ";
        stream << fixed << setprecision(4) << minDistance;
        string text = stream.str();
        cv::putText(im, text, p1, FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 0, 0), 2);
      }

      // Show result
      cv::imshow("webcam", im);
      int k = cv::waitKey(1);
      // Quit when Esc is pressed
      if (k == 27) {
        break;
      }
    }
  // Counter used for skipping frames
  count += 1;
  }
  cv::destroyAllWindows();
}
