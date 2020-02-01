#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <map>

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



static void listdir(string dirName, std::vector<string>& folderNames, std::vector<string>& fileNames, std::vector<string>& symlinkNames) {
  DIR *dir;
  struct dirent *ent;

  if ((dir = opendir(dirName.c_str())) != NULL) {
    while ((ent = readdir(dir)) != NULL) {

      if((strcmp(ent->d_name,".") == 0) || (strcmp(ent->d_name,"..") == 0)) {
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


static void filterFiles(string dirPath, std::vector<string>& fileNames, std::vector<string>& filteredFilePaths, string ext, std::vector<int>& imageLabels, int index)
{
  for(int i = 0; i < fileNames.size(); i++) {
    string fname = fileNames[i];
    if (fname.find(ext, (fname.length() - ext.length())) != std::string::npos) {
      filteredFilePaths.push_back(dirPath + "/" + fname);
      imageLabels.push_back(index);
    }
  }
}

template<typename T>
static void printVector(std::vector<T>& vec) {
  for (int i = 0; i < vec.size(); i++) {
    cout << i << " " << vec[i] << "; ";
  }
  cout << endl;
}

int opencv_enrollment() {
  
  const std::string recModelPath = "C:/Users/xwen2/Desktop/Projects/2. Face Recognition/data/models/openface.nn4.small2.v1.t7";
  frontal_face_detector faceDetector = get_frontal_face_detector();
  dnn::Net recModel = dnn::readNetFromTorch(recModelPath);
  dlib::shape_predictor landmarkDetector;
  dlib::deserialize("C:/Users/xwen2/Desktop/Projects/2. Face Recognition/data/models/shape_predictor_5_face_landmarks.dat") >> landmarkDetector;


  string faceDatasetFolder = "C:/Users/xwen2/Desktop/Projects/2. Face Recognition/data/images/faces";
  std::vector<string> subfolders, fileNames, symlinkNames;
  
  listdir(faceDatasetFolder, subfolders, fileNames, symlinkNames);

  // names: vector containing names of subfolders i.e. persons
  // labels: integer labels assigned to persons
  // labelNameMap: dict containing (integer label, person name) pairs
  std::vector<string> names;
  std::vector<int> labels;
  std::map<int, string> labelNameMap;

  names.push_back("unknown");
  labels.push_back(-1);

  // imagePaths: vector containing imagePaths
  // imageLabels: vector containing integer labels corresponding to imagePaths
  std::vector<string> imagePaths;
  std::vector<int> imageLabels;

  
  std::vector<string> folderNames;

  for (int i = 0; i < subfolders.size(); i++)
  {
    string personFolderName = subfolders[i];
    std::size_t found = personFolderName.find_last_of("/\\");
    string name = personFolderName.substr(found+1);
    
    int label = i;
    
    names.push_back(name);
    labels.push_back(label);
   
    labelNameMap[label] = name;

    folderNames.clear();
    fileNames.clear();
    symlinkNames.clear();

    listdir(subfolders[i], folderNames, fileNames, symlinkNames);

    filterFiles(subfolders[i], fileNames, imagePaths, "jpg", imageLabels, i);
    }


  std::vector<Mat> faceDescriptors;
  std::vector<int> faceLabels;
  Mat faceDescriptor;

  for (int i = 0; i < imagePaths.size(); i++)
  {
    string imagePath = imagePaths[i];
    int imageLabel = imageLabels[i];

    cout << "processing: " << imagePath << endl;


    Mat im = cv::imread(imagePath);

    cv_image<bgr_pixel> imDlib(im);
    std::vector<dlib::rectangle> faceRects = faceDetector(imDlib);
    cout << faceRects.size() << " Face(s) Found" << endl;

    for (int j = 0; j < faceRects.size(); j++) {
      Mat alignedFace;
      alignFace(im, alignedFace, faceRects[j], landmarkDetector, cv::Size(96, 96));

      cv::Mat blob = dnn::blobFromImage(alignedFace, 1.0/255, cv::Size(96, 96), Scalar(0,0,0), false, false);
      recModel.setInput(blob);
      faceDescriptor = recModel.forward();


      faceDescriptors.push_back(faceDescriptor.clone());

      faceLabels.push_back(imageLabel);

    }
  }
  cout << "number of face descriptors " << faceDescriptors.size() << endl;
  cout << "number of face labels " << faceLabels.size() << endl;


  const string labelNameFile = "label_name_openface.txt";
  ofstream of;
  of.open (labelNameFile);
  for (int m = 0; m < names.size(); m++) {
    of << names[m];
    of << ";";
    of << labels[m];
    of << "\n";
  }
  of.close();


  const string descriptorsPath = "descriptors_openface.csv";
  ofstream ofs;
  ofs.open(descriptorsPath);

  for (int m = 0; m < faceDescriptors.size(); m++) {
    Mat faceDescriptorVec = faceDescriptors[m];
    ofs << faceLabels[m];
    ofs << ";";
    for (int n = 0; n < faceDescriptorVec.cols; n++) {
      ofs << std::fixed << std::setprecision(8) << faceDescriptorVec.at<float>(n);
      
      if ( n == (faceDescriptorVec.cols - 1)) {
        ofs << "\n";  
      } else {
        ofs << ";"; 
      }
    }
  }
  ofs.close();
  return 1;
}
