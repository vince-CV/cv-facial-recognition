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

 #include <dlib/string.h>
 #include <dlib/dnn.h>
 #include <dlib/image_io.h>
 #include <dlib/opencv.h>
 #include <dlib/image_processing.h>
 #include <dlib/image_processing/frontal_face_detector.h>


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

#define THRESHOLD 0.5


template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                            input_rgb_image_sized<150>
                            >>>>>>>>>>>>;

template<typename T>
static void printVector(std::vector<T>& vec) {
  for (int i = 0; i < vec.size(); i++) {
    cout << i << " " << vec[i] << "; ";
  }
  cout << endl;
}


static void readLabelNameMap(const string& filename, std::vector<string>& names, std::vector<int>& labels,
                             std::map<int, string>& labelNameMap, char separator = ';') {
  std::ifstream file(filename.c_str(), ifstream::in);
  if (!file) {
    string error_message = "No valid input file was given, please check the given filename.";
    CV_Error(CV_StsBadArg, error_message);
  }
  string line;
  string name, labelStr;

  while (getline(file, line)) {
    stringstream liness(line);

    getline(liness, name, separator);
   
    getline(liness, labelStr);
    if(!name.empty() && !labelStr.empty()) {
      names.push_back(name);
      
      int label = atoi(labelStr.c_str());
      labels.push_back(label);
     
      labelNameMap[label] = name;
    }
  }
}


static void readDescriptors(const string& filename, std::vector<int>& faceLabels, std::vector<matrix<float,0,1>>& faceDescriptors, char separator = ';') {
  std::ifstream file(filename.c_str(), ifstream::in);
  if (!file) {
    string error_message = "No valid input file was given, please check the given filename.";
    CV_Error(CV_StsBadArg, error_message);
  }

  string line;
  string faceLabel;

  string valueStr;
  float value;
  std::vector<float> faceDescriptorVec;

  while (getline(file, line)) 
  {
    stringstream liness(line);

    getline(liness, faceLabel, separator);
    if(!faceLabel.empty()) {
      faceLabels.push_back(std::atoi(faceLabel.c_str()));
    }

    faceDescriptorVec.clear();

    while (getline(liness, valueStr, separator)) {
      if (!valueStr.empty()) {

        faceDescriptorVec.push_back(atof(valueStr.c_str()));
      }
    }


    dlib::matrix<float, 0, 1> faceDescriptor = dlib::mat(faceDescriptorVec);
    faceDescriptors.push_back(faceDescriptor);
  }
}


static void nearestNeighbor(dlib::matrix<float, 0, 1>& faceDescriptorQuery,std::vector<dlib::matrix<float, 0, 1>>& faceDescriptors,std::vector<int>& faceLabels, int& label, float& minDistance) 
{
  int minDistIndex = 0;
  minDistance = 1.0;
  label = -1;

  for (int i = 0; i < faceDescriptors.size(); i++) 
  {
    double distance = length(faceDescriptors[i] - faceDescriptorQuery);
    if (distance < minDistance) 
	{
      minDistance = distance;
      minDistIndex = i;
    }
  }
 
  if (minDistance > THRESHOLD)
  {
    label = -1;
  } 
  else 
  {
    label = faceLabels[minDistIndex];
  }
}

int dlib_inference_m(int argc, char *argv[]) {
  
  String predictorPath, faceRecognitionModelPath;
  predictorPath = "C:/Users/xwen2/Desktop/Projects/2. Face Recognition/data/models/shape_predictor_68_face_landmarks.dat";
  faceRecognitionModelPath = "C:/Users/xwen2/Desktop/Projects/2. Face Recognition/data/models/dlib_face_recognition_resnet_model_v1.dat";
  frontal_face_detector faceDetector = get_frontal_face_detector();
  shape_predictor landmarkDetector;
  deserialize(predictorPath) >> landmarkDetector;
  anet_type net;
  deserialize(faceRecognitionModelPath) >> net;

  std::map<int, string> labelNameMap;
  std::vector<string> names;
  std::vector<int> labels;
  const string labelNameFile = "label_name.txt";
  readLabelNameMap(labelNameFile, names, labels, labelNameMap);

  const string faceDescriptorFile = "descriptors.csv";
  std::vector<int> faceLabels;
  std::vector<matrix<float,0,1>> faceDescriptors;
  readDescriptors(faceDescriptorFile, faceLabels, faceDescriptors);

  string imagePath;
  if (argc > 1) 
  {
    imagePath = argv[1];
  } 
  else 
  {
    imagePath = "C:/Users/xwen2/Desktop/Projects/2. Face Recognition/data/images/faces/satya_demo.jpg";
  }
  Mat im = cv::imread(imagePath, cv::IMREAD_COLOR);

  if (im.empty())
  {
    exit(0);
  }
  double t = cv::getTickCount();

  Mat imRGB = im.clone();
  cv::cvtColor(im, imRGB, cv::COLOR_BGR2RGB);

  dlib::matrix<dlib::rgb_pixel> imDlib(dlib::mat(dlib::cv_image<dlib::rgb_pixel>(imRGB)));

 
  std::vector<dlib::rectangle> faceRects = faceDetector(imDlib);
  cout << faceRects.size() << " Faces Detected " << endl;
  string name;

  for (int i = 0; i < faceRects.size(); i++) 
  {

    full_object_detection landmarks = landmarkDetector(imDlib, faceRects[i]);

    matrix<rgb_pixel> face_chip;

    extract_image_chip(imDlib, get_face_chip_details(landmarks,150,0.25), face_chip);

    matrix<float,0,1> faceDescriptorQuery = net(face_chip);

    int label;
    float minDistance;
    nearestNeighbor(faceDescriptorQuery, faceDescriptors, faceLabels, label, minDistance);
    name = labelNameMap[label];

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
  cv::imwrite(cv::format("output-dlib-%s.jpg",name.c_str()),im);
  int k = cv::waitKey(0);

  cv::destroyAllWindows();
}
