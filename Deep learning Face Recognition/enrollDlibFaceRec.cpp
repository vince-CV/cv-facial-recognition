

#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <map>

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
static void printVector(std::vector<T>& vec) 
{
  for (int i = 0; i < vec.size(); i++) {
    cout << i << " " << vec[i] << "; ";
  }
  cout << endl;
}

int dlib_enrollment() {

  String predictorPath, faceRecognitionModelPath;
  predictorPath = "C:/Users/xwen2/Desktop/Projects/2. Face Recognition/data/models/shape_predictor_68_face_landmarks.dat";
  faceRecognitionModelPath = "C:/Users/xwen2/Desktop/Projects/2. Face Recognition/data/models/dlib_face_recognition_resnet_model_v1.dat";
  frontal_face_detector faceDetector = get_frontal_face_detector();
  shape_predictor landmarkDetector;
  deserialize(predictorPath) >> landmarkDetector;
  anet_type net;
  deserialize(faceRecognitionModelPath) >> net;


  string faceDatasetFolder = "C:/Users/xwen2/Desktop/Projects/2. Face Recognition/data/images/faces";
  std::vector<string> subfolders, fileNames, symlinkNames;

  listdir(faceDatasetFolder, subfolders, fileNames, symlinkNames);

  std::vector<string> names;
  std::vector<int> labels;
  std::map<int, string> labelNameMap;

  names.push_back("unknown");
  labels.push_back(-1);

  std::vector<string> imagePaths;
  std::vector<int> imageLabels;


  std::vector<string> folderNames;

  for (int i = 0; i < subfolders.size(); i++) {
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

  std::vector<matrix<float,0,1>> faceDescriptors;
  std::vector<int> faceLabels;


  for (int i = 0; i < imagePaths.size(); i++) 
  {
    string imagePath = imagePaths[i];
    int imageLabel = imageLabels[i];

    cout << "processing: " << imagePath << endl;

    Mat im = cv::imread(imagePath, cv::IMREAD_COLOR);

    Mat imRGB;
    cv::cvtColor(im, imRGB, cv::COLOR_BGR2RGB);

    dlib::matrix<dlib::rgb_pixel> imDlib(dlib::mat(dlib::cv_image<dlib::rgb_pixel>(imRGB)));

    std::vector<dlib::rectangle> faceRects = faceDetector(imDlib);
    cout << faceRects.size() << " Face(s) Found" << endl;

    for (int j = 0; j < faceRects.size(); j++) {

      full_object_detection landmarks = landmarkDetector(imDlib, faceRects[j]);

      matrix<rgb_pixel> face_chip;


      extract_image_chip(imDlib, get_face_chip_details(landmarks, 150, 0.25), face_chip);

      matrix<float,0,1> faceDescriptor = net(face_chip);

      faceDescriptors.push_back(faceDescriptor);

      faceLabels.push_back(imageLabel);
    }
  }

  cout << "number of face descriptors " << faceDescriptors.size() << endl;
  cout << "number of face labels " << faceLabels.size() << endl;


  const string labelNameFile = "label_name.txt";
  ofstream of;
  of.open (labelNameFile);
  for (int m = 0; m < names.size(); m++) {
    of << names[m];
    of << ";";
    of << labels[m];
    of << "\n";
  }
  of.close();


  const string descriptorsPath = "descriptors_dlib.csv";
  ofstream ofs;
  ofs.open(descriptorsPath);

  for (int m = 0; m < faceDescriptors.size(); m++) {
    matrix<float,0,1> faceDescriptor = faceDescriptors[m];
    std::vector<float> faceDescriptorVec(faceDescriptor.begin(), faceDescriptor.end());

    ofs << faceLabels[m];
    ofs << ";";
    for (int n = 0; n < faceDescriptorVec.size(); n++) {
      ofs << std::fixed << std::setprecision(8) << faceDescriptorVec[n];

      if ( n == (faceDescriptorVec.size() - 1)) {
        ofs << "\n";  
      } else {
        ofs << ";";  
      }
    }
  }
  ofs.close();
  return 1;
}
