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
using namespace cv::face;
using namespace std;


#define faceWidth 64
#define faceHeight 64

#define PI 3.14159265

static void listdir(string dirName, vector<string>& folderNames, vector<string>& fileNames, vector<string>& symlinkNames)
{
  DIR *dir;
  struct dirent *ent;

  if ((dir = opendir(dirName.c_str())) != NULL) {
    
    while ((ent = readdir(dir)) != NULL) {
 
      if(strcmp(ent->d_name,".") == 0 || strcmp(ent->d_name,"..") == 0 ) {
      continue;
      }
      string temp_name = ent->d_name;
      
      switch (ent->d_type) {
        case DT_REG:
          fileNames.push_back(temp_name);
          break;
        case DT_DIR:
          folderNames.push_back(dirName + "/" + temp_name);
          break;
        case DT_LNK:
          symlinkNames.push_back(temp_name);
          break;
        default:
          break;
      }
      
    }
 
    std::sort(folderNames.begin(), folderNames.end());
    std::sort(fileNames.begin(), fileNames.end());
    std::sort(symlinkNames.begin(), symlinkNames.end());
    closedir(dir);
  }
}


static void filterFiles(string dirPath, vector<string>& fileNames, vector<string>& filteredFilePaths, string ext)
{
  for(int i = 0; i < fileNames.size(); i++)
  {
    string fname = fileNames[i];
    if (fname.find(ext, (fname.length() - ext.length())) != std::string::npos)
    {
      filteredFilePaths.push_back(dirPath + "/" + fname);
    }
  }
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

static void printVector(vector<string>& vec)
{
  for (int i = 0; i < vec.size(); i++)
  {
    cout << vec[i] << endl;
  }
}

static Mat getCroppedFaceRegion(Mat image, std::vector<Point2f> landmarks)
{
    int imWidth = image.cols;
    int imHeight = image.rows;
    int x1Limit = landmarks[0].x - (landmarks[36].x - landmarks[0].x);
    int x2Limit = landmarks[16].x + (landmarks[16].x - landmarks[45].x);
    int y1Limit = landmarks[27].y - 3*(landmarks[30].y - landmarks[27].y);
    int y2Limit = landmarks[8].y + (landmarks[30].y - landmarks[29].y);

    int x1 = max(x1Limit,0);
    int x2 = min(x2Limit, imWidth);
    int y1 = max(y1Limit, 0);
    int y2 = min(y2Limit, imHeight);

    Mat cropped;
    cv::Rect selectedRegion = cv::Rect( x1, y1, x2-x1, y2-y1 );
    cropped = image(selectedRegion);
    return cropped;
}

int traditional_train()
{
  string faceDatasetFolder = "C:/Users/xwen2/Desktop/Projects/2. Face Recognition/data/images/FaceRec/trainFaces";
  vector<string> faceFolderNames, fileNames, symlinkNames;

  listdir(faceDatasetFolder, faceFolderNames, fileNames, symlinkNames);

  vector<string> names;
  vector<int> labels;
  map<int, string> labelNameMap;
  names.push_back("unknown");
  labels.push_back(-1);

  vector<string> imagePaths;
  vector<int> imageLabels;

  vector<string> folderNames;
  for (int i = 0; i < faceFolderNames.size(); i++)
  {
    string faceFolderName = faceFolderNames[i];
    std::size_t found = faceFolderName.find_last_of("/\\");
    string name = faceFolderName.substr(found+1);
    int label = i;
    names.push_back(name);
    labels.push_back(label);
    labelNameMap[label] = name;

    folderNames.clear();
    fileNames.clear();
    symlinkNames.clear();

    listdir(faceFolderName, folderNames, fileNames, symlinkNames);

    filterFiles(faceFolderName, fileNames, imagePaths, "jpg");
    filterFiles(faceFolderName, fileNames, imagePaths, "pgm");

    for (int j = 0; j < fileNames.size(); j++)
    {
      imageLabels.push_back(i);
    }
  }

  dlib::frontal_face_detector faceDetector = dlib::get_frontal_face_detector();

  dlib::shape_predictor landmarkDetector;
  dlib::deserialize("C:/Users/xwen2/Desktop/Projects/2. Face Recognition/data/models/shape_predictor_68_face_landmarks.dat") >> landmarkDetector;

  vector<Mat> imagesFaceTrain;
  vector<int> labelsFaceTrain;
  cout << "Training . . ." << endl;

  for (int i = 0; i < imagePaths.size(); i++)
  {
    String imagePath = imagePaths[i];
    cout << imagePath << endl;

    Mat im = cv::imread(imagePath);

    std::vector<Point2f> landmarks = getLandmarks(faceDetector, landmarkDetector, im);
    if(landmarks.size() < 68)
    {
      cout << "only " << landmarks.size() << " landmarks found" << endl;
      continue;
    }

    cvtColor(im, im, COLOR_BGR2GRAY);
    Mat imFace = getCroppedFaceRegion(im, landmarks);

    Mat alignedImFace;
    alignFace(imFace, alignedImFace, landmarks);
    cv::resize(alignedImFace, alignedImFace, Size(faceHeight, faceWidth));

    alignedImFace.convertTo(alignedImFace, CV_32F, 1.0/255);

    imagesFaceTrain.push_back(alignedImFace);
    labelsFaceTrain.push_back(imageLabels[i]);

    if(i==330)
    {
      alignedImFace.convertTo(alignedImFace, CV_8UC3, 255);
     // imwrite("Image.jpg",im);
     // imwrite("Cropped_Face.jpg", imFace);
     // imwrite("Aligned_Face.jpg", alignedImFace);
    }
  }

  cout << "Training using Eigen Faces, model saved" << endl;
  Ptr<FaceRecognizer> faceRecognizerEigen = EigenFaceRecognizer::create();
  faceRecognizerEigen->train(imagesFaceTrain, labelsFaceTrain);
  faceRecognizerEigen->write("face_model_eigen.yml");

  cout << "Training using Fisher Faces, model saved" << endl;
  Ptr<FaceRecognizer> faceRecognizerFisher = FisherFaceRecognizer::create();
  faceRecognizerFisher->train(imagesFaceTrain, labelsFaceTrain);
  faceRecognizerFisher->write("face_model_fisher.yml");

  cout << "Training using LBPH, model saved" << endl;
  Ptr<FaceRecognizer> faceRecognizerLBPH = LBPHFaceRecognizer::create();
  faceRecognizerLBPH->train(imagesFaceTrain, labelsFaceTrain);
  faceRecognizerLBPH->write("face_model_lbph.yml");


  const string labelNameFile = "labels_map.txt";
  ofstream of;
  of.open (labelNameFile);
  for (int m = 0; m < names.size(); m++) {
    of << names[m];
    of << ";";
    of << labels[m];
    of << "\n";
  }
  of.close();

  return 0;
}
