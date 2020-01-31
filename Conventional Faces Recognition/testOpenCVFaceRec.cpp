#include <opencv2/core.hpp>
#include <opencv2/face.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <map>
#include "faceBlendCommon.hpp"

#ifdef _WIN32
  #include "dirent.h"
#elif __APPLE__
  #include "TargetConditionals.h"
  #if TARGET_OS_MAC
    #include <dirent.h>
  #else
    #error "Not Mac. Find al alternative to dirent"
  #endif
#elif __linux__
  #include <dirent.h>
#elif __unix__ 
  #include <dirent.h>
#else
  #error "Unknown compiler"
#endif

using namespace cv;
using namespace cv::face;
using namespace std;

#define faceWidth 64
#define faceHeight 64

#define PI 3.14159265
#define RESIZE_HEIGHT 480

#define VIDEO 0


// 'l' = LBPH 'f' = Fisher 'e' = Eigen 
#define MODEL 'l'

static Mat getCroppedFaceRegion(Mat image, std::vector<Point2f> landmarks, cv::Rect &selectedRegion)
{
  int x1Limit = landmarks[0].x - (landmarks[36].x - landmarks[0].x);
  int x2Limit = landmarks[16].x + (landmarks[16].x - landmarks[45].x);
  int y1Limit = landmarks[27].y - 3*(landmarks[30].y - landmarks[27].y);
  int y2Limit = landmarks[8].y + (landmarks[30].y - landmarks[29].y);

  int imWidth = image.cols;
  int imHeight = image.rows;
  int x1 = max(x1Limit,0);
  int x2 = min(x2Limit, imWidth);
  int y1 = max(y1Limit, 0);
  int y2 = min(y2Limit, imHeight);


  Mat cropped;
  selectedRegion = cv::Rect( x1, y1, x2-x1, y2-y1 );
  cropped = image(selectedRegion);
  return cropped;
}

static void alignFace(Mat &imFace, Mat &alignedImFace, std::vector<Point2f> landmarks)
{
  float l_x = landmarks[39].x;
  float l_y = landmarks[39].y;
  float r_x = landmarks[42].x;
  float r_y = landmarks[42].y;

  float dx = r_x - l_x;
  float dy = r_y - l_y;
  double angle = atan2(dy, dx) * 180 / PI;

  Point2f eyesCenter;
  eyesCenter.x = (l_x + r_x) / 2.0;
  eyesCenter.y = (l_y + r_y) / 2.0;

  Mat rotMatrix = Mat(2, 3, CV_32F);
  rotMatrix = getRotationMatrix2D(eyesCenter, angle, 1);
  warpAffine(imFace, alignedImFace, rotMatrix, imFace.size());
}

static void getFileNames(string dirName, vector<string> &imageFnames)
{
  DIR *dir;
  struct dirent *ent;
  int count = 0;

  string imgExt1 = "pgm";
  string imgExt2 = "jpg";

  vector<string> files;

  if ((dir = opendir (dirName.c_str())) != NULL)
  {
    while ((ent = readdir (dir)) != NULL)
    {
      if(strcmp(ent->d_name,".") == 0 | strcmp(ent->d_name, "..") == 0)
      {
        continue;
      }
      string temp_name = ent->d_name;
      files.push_back(temp_name);
    }

    std::sort(files.begin(),files.end());
    for(int it=0;it<files.size();it++)
    {
      string path = dirName;
      string fname=files[it];

      if (fname.find(imgExt1, (fname.length() - imgExt1.length())) != std::string::npos)
      {
        path.append(fname);
        imageFnames.push_back(path);
      }
      else if (fname.find(imgExt2, (fname.length() - imgExt2.length())) != std::string::npos)
      {
        path.append(fname);
        imageFnames.push_back(path);
      }
    }
    closedir (dir);
  }
}

static void readLabelNameMap(const string& filename, vector<string>& names, vector<int>& labels, map<int, string>& labelNameMap, char separator = ';')
{
  std::ifstream file(filename.c_str(), ifstream::in);
  if (!file)
  {
    string error_message = "No valid input file was given, please check the given filename.";
    CV_Error(CV_StsBadArg, error_message);
  }
  string line;
  string name, classlabel;
  while (getline(file, line))
  {

    stringstream liness(line);
    getline(liness, name, separator);
    getline(liness, classlabel);

    if(!name.empty() && !classlabel.empty()) {
      names.push_back(name);
      int label = atoi(classlabel.c_str());
      labels.push_back(label);
      labelNameMap[label] = name;
    }
  }
}

int traditional_test(int argc, char** argv)
{
  cv::VideoCapture cap;
  vector<string> testFiles;
  int testFileCount = 0;

  if(VIDEO)
  {
    cap.open("C:/Users/xwen2/Desktop/Projects/2. Face Recognition/data/videos/face1.mp4");

    if (!cap.isOpened())
	{
      cerr << "Unable to connect to camera" << endl;
      return 1;
    }
  }
  else
  {
    string testDatasetFolder = "C:/Users/xwen2/Desktop/Projects/2. Face Recognition/data/images/FaceRec/testFaces/";
    getFileNames(testDatasetFolder, testFiles);
    testFileCount = 0;
  }

  Ptr<FaceRecognizer> faceRecognizer;
  if(MODEL == 'e')
  {
    cout << "Using Eigen Faces" << endl;
    faceRecognizer = EigenFaceRecognizer::create();
    faceRecognizer->read("face_model_eigen.yml");
  }
  else if(MODEL == 'f')
  {
    cout << "Using Fisher Faces" << endl;
    faceRecognizer = FisherFaceRecognizer::create();
    faceRecognizer->read("face_model_fisher.yml");
  }
  else if(MODEL == 'l')
  {
    cout << "Using LBPH" << endl;
    faceRecognizer = LBPHFaceRecognizer::create();
    faceRecognizer->read("face_model_lbph.yml");
  }


  map<int, string> labelNameMap;
  vector<string> names;
  vector<int> labels;
  const string labelFile = "labels_map.txt";
  readLabelNameMap(labelFile, names, labels, labelNameMap);


  dlib::frontal_face_detector faceDetector = dlib::get_frontal_face_detector();

 
  dlib::shape_predictor landmarkDetector;
  dlib::deserialize("C:/Users/xwen2/Desktop/Projects/2. Face Recognition/data/models/shape_predictor_68_face_landmarks.dat") >> landmarkDetector;


  float IMAGE_RESIZE;
  Mat im, imGray;

  while (1)
  {
    
    if( VIDEO )
    {
      cap >> im;
    }

    else
    {
      im = imread(testFiles[testFileCount]);
      testFileCount++;
    }

    if (im.empty())
      break;

    IMAGE_RESIZE = (float)im.rows/RESIZE_HEIGHT;
    cv::resize(im, im, cv::Size(), 1.0/IMAGE_RESIZE, 1.0/IMAGE_RESIZE);

    std::vector<Point2f> landmarks = getLandmarks(faceDetector, landmarkDetector, im);
    if(landmarks.size() < 68)
    {
      cout << "Only " << landmarks.size() << " landmarks found, continuing with next frame" << endl;
      continue;
    }

    cvtColor(im, imGray, COLOR_BGR2GRAY);

    cv::Rect faceRegion;
    Mat imFace = getCroppedFaceRegion(imGray, landmarks, faceRegion);

    Mat alignedImFace;
    alignFace(imFace, alignedImFace, landmarks);
    cv::resize(alignedImFace, alignedImFace, Size(faceHeight, faceWidth));
    alignedImFace.convertTo(alignedImFace, CV_32F, 1.0/255);

    int predictedLabel = -1;
    double score = 0.0;
    faceRecognizer->predict(alignedImFace, predictedLabel, score);

    Point2d center = Point2d(faceRegion.x + faceRegion.width/2.0, faceRegion.y + faceRegion.height/2.0);
    int radius = static_cast<int>(faceRegion.height/2.0);

    cv::circle(im, center, radius, Scalar(0, 255, 0), 1, LINE_8);

    cv::putText(im, labelNameMap[predictedLabel], Point(10,100), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 0, 0), 2);

    cv::imshow("Face Recognition demo", im);

    int k = 0;
    if(VIDEO)
      k = cv::waitKey(10);
    else
      k = cv::waitKey(1000);

    if (k == 27)
    {
      break;
    }
  }

  if(VIDEO)
    cap.release();
  cv::destroyAllWindows();

  return 0;
}
