

 #include <iostream>
 #include <fstream>
 #include <sstream>
 #include <math.h>
 #include <map>
 #include <iomanip> 
 #include <sstream> 

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
#elif __unix__ 
  #include <dirent.h>
#else
  #error "Unknown compiler"
#endif

using namespace cv;
using namespace dlib;
using namespace std;

#define SKIP_FRAMES 1
#define recThreshold 0.8



template<typename T>
static void printVector(std::vector<T>& vec) {
  for (int i = 0; i < vec.size(); i++) {
    cout << i << " " << vec[i] << "; ";
  }
  cout << endl;
}


static void readLabelNameMap(const string& filename, std::vector<string>& names, std::vector<int>& labels, std::map<int, string>& labelNameMap, char separator = ';') 
{
  std::ifstream file(filename.c_str(), ifstream::in);
  if (!file) {
    string error_message = "No valid input file was given, please check the given filename.";
    CV_Error(CV_StsBadArg, error_message);
  }
  string line;
  string name, labelStr;

  while (getline(file, line)) 
  {
    stringstream liness(line);

    getline(liness, name, separator);

    getline(liness, labelStr);
    if(!name.empty() && !labelStr.empty()) 
	{
      names.push_back(name);
  
      int label = atoi(labelStr.c_str());
      labels.push_back(label);
    
      labelNameMap[label] = name;
    }
  }
}


static void readDescriptors(const string& filename, std::vector<int>& faceLabels, std::vector<Mat>& faceDescriptors, char separator = ';') 
{
  std::ifstream file(filename.c_str(), ifstream::in);
  if (!file) 
  {
    string error_message = "No valid input file was given, please check the given filename.";
    CV_Error(CV_StsBadArg, error_message);
  }

  string line;
  string faceLabel;

  string valueStr;
  float value;
  std::vector<float> faceDescriptorVec;

  while (getline(file, line)) {
    stringstream liness(line);

    getline(liness, faceLabel, separator);
    if(!faceLabel.empty()) {
      faceLabels.push_back(std::atoi(faceLabel.c_str()));
    }

    faceDescriptorVec.clear();

    while (getline(liness, valueStr, separator)) 
	{
      if (!valueStr.empty()) 
	  {

        faceDescriptorVec.push_back(atof(valueStr.c_str()));
      }
    }


    Mat faceDescriptor(faceDescriptorVec);
    faceDescriptors.push_back(faceDescriptor.clone());
  }
}


static void nearestNeighbor(Mat& faceDescriptorQuery,std::vector<Mat>& faceDescriptors,std::vector<int>& faceLabels, int& label, float& minDistance) 
{
  int minDistIndex = 0;
  minDistance = 1.0;
  label = -1;

  for (int i = 0; i < faceDescriptors.size(); i++) {
    double distance = cv::norm(faceDescriptors[i].t() - faceDescriptorQuery);
    if (distance < minDistance) {
      minDistance = distance;
      minDistIndex = i;
    }
  }
 
  if (minDistance > recThreshold)
  {
    label = -1;
  } else {
    label = faceLabels[minDistIndex];
  }
}

int cv_inference_v(int argc, const char** argv) {


  const std::string recModelPath = "C:/Users/xwen2/Desktop/Projects/2. Face Recognition/data/models/openface.nn4.small2.v1.t7";
  frontal_face_detector faceDetector = get_frontal_face_detector();
  dnn::Net recModel = dnn::readNetFromTorch(recModelPath);
  dlib::shape_predictor landmarkDetector;
  dlib::deserialize("C:/Users/xwen2/Desktop/Projects/2. Face Recognition/data/models/shape_predictor_5_face_landmarks.dat") >> landmarkDetector;



  std::map<int, string> labelNameMap;
  std::vector<string> names;
  std::vector<int> labels;
  const string labelNameFile = "label_name_openface.txt";
  readLabelNameMap(labelNameFile, names, labels, labelNameMap);


  const string faceDescriptorFile = "descriptors_openface.csv";
  std::vector<int> faceLabels;
  std::vector<Mat> faceDescriptors;
  readDescriptors(faceDescriptorFile, faceLabels, faceDescriptors);
  printVector(faceDescriptors);


  VideoCapture cap;
  cap.open("C:/Users/xwen2/Desktop/Projects/2. Face Recognition/data/videos/face1.mp4");

  if (!cap.isOpened()) 
  {
    cerr << "Unable to connect to camera" << endl;
    return 1;
  }

  int count = 0;
  double t = cv::getTickCount();

  while (1) 
  {
    t = cv::getTickCount();

    Mat im;
    cap >> im;


    if (im.empty()){
      break;
    }


    if ((count % SKIP_FRAMES) == 0) {

      cv_image<bgr_pixel> imDlib(im);


      std::vector<dlib::rectangle> faceRects = faceDetector(imDlib);

      for (int i = 0; i < faceRects.size(); i++) 
	  {
        cout << faceRects.size() << " Face(s) Found" << endl;

        Mat alignedFace;
        alignFace(im, alignedFace, faceRects[i], landmarkDetector, cv::Size(96, 96));
        cv::Mat blob = dnn::blobFromImage(alignedFace, 1.0/255, cv::Size(96, 96), Scalar(0,0,0), false, false);
        recModel.setInput(blob);
        Mat faceDescriptorQuery = recModel.forward();


        int label;
        float minDistance;
        nearestNeighbor(faceDescriptorQuery, faceDescriptors, faceLabels, label, minDistance);

        string name = labelNameMap[label];

        cout << "Time taken = " << ((double)cv::getTickCount() - t)/cv::getTickFrequency() << endl;

        Point2d p1 = Point2d(faceRects[i].left(), faceRects[i].top());
        Point2d p2 = Point2d(faceRects[i].right(), faceRects[i].bottom());
        cv::rectangle(im, p1, p2, Scalar(0, 0, 255), 1, LINE_8);

        Point2d center = Point((faceRects[i].left() + faceRects[i].right())/2.0,
        (faceRects[i].top() + faceRects[i].bottom())/2.0 );
        int radius = static_cast<int> ((faceRects[i].bottom() - faceRects[i].top())/2.0);
        cv::circle(im, center, radius, Scalar(0, 255, 0), 1, LINE_8);

        stringstream stream;
        stream << name << " ";
        stream << fixed << setprecision(4) << minDistance;
        string text = stream.str();
        cv::putText(im, text, p1, FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 0, 0), 2);
      }

      cv::imshow("webcam", im);
      int k = cv::waitKey(1);

      if (k == 27) 
	  {
        break;
      }
    }

  count += 1;
  }
  cv::destroyAllWindows();
}
