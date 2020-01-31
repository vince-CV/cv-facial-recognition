#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/core/core.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "dirent.h"
#include <stdlib.h>
#include <time.h>

using namespace cv;
using namespace std;

#define MAX_SLIDER_VALUE 255
#define NUM_EIGEN_FACES 10

int sliderValues[NUM_EIGEN_FACES];

Mat averageFace;
vector<Mat> eigenFaces;


void readImages(string dirName, vector<Mat> &images)
{

  cout << "Reading images from " << dirName;

  if (!dirName.empty() && dirName.back() != '/')
    dirName += '/';

  DIR *dir;
  struct dirent *ent;
  int count = 0;

  string imgExt = "jpg";
  vector<string> files;

  if ((dir = opendir (dirName.c_str())) != NULL)
  {
    while ((ent = readdir (dir)) != NULL)
    {
      if(strcmp(ent->d_name,".") == 0 || strcmp(ent->d_name,"..") == 0 )
      {
        continue;
      }
      string fname = ent->d_name;

      if (fname.find(imgExt, (fname.length() - imgExt.length())) != std::string::npos)
      {
        string path = dirName + fname;
        Mat img = imread(path);
        if(!img.data)
        {
          cout << "image " << path << " not read properly" << endl;
        }
        else
        {

          img.convertTo(img, CV_32FC3, 1/255.0);
          images.push_back(img);

          Mat imgFlip;
          flip(img, imgFlip, 1);
          images.push_back(imgFlip);
        }
      }
    }
    closedir (dir);
  }


  if(images.empty())exit(EXIT_FAILURE);

  cout << "... " << images.size() / 2 << " files read"<< endl;

}


static  Mat createDataMatrix(const vector<Mat> &images)
{
  cout << "Creating data matrix from images ...";

  Mat data(static_cast<int>(images.size()), images[0].rows * images[0].cols * 3, CV_32F);

  for(unsigned int i = 0; i < images.size(); i++)
  {
    Mat image = images[i].reshape(1,1);
    image.copyTo(data.row(i));
  }

  cout << " DONE" << endl;
  return data;
}

void createNewFace(int ,void *)
{

  Mat output = averageFace.clone();

  for(int i = 0; i < NUM_EIGEN_FACES; i++)
  {

    double weight = sliderValues[i] - MAX_SLIDER_VALUE/2;
    output = output + eigenFaces[i] * weight;
  }

  resize(output, output, Size(), 2, 2);

  imshow("Result", output);

}


void resetSliderValues(int event, int x, int y, int flags, void* userdata)
{
  if (event == EVENT_LBUTTONDOWN)
  {
    for(int i = 0; i < NUM_EIGEN_FACES; i++)
    {
      sliderValues[i] = 128;
      setTrackbarPos("Weight" + to_string(i), "Trackbars", MAX_SLIDER_VALUE/2);
    }

    createNewFace(0,0);

  }
}


int eigen(int argc, char **argv)
{
 
  string dirName = "C:/Users/xwen2/Desktop/Projects/2. Face Recognition/data/images/eigenface/";

  vector<Mat> images;
  readImages(dirName, images);

  Size sz = images[0].size();


  Mat data = createDataMatrix(images);

  cout << "Calculating PCA ...";
  PCA pca(data, Mat(), PCA::DATA_AS_ROW, NUM_EIGEN_FACES);
  cout << " DONE"<< endl;

  averageFace = pca.mean.reshape(3,sz.height);

  Mat eigenVectors = pca.eigenvectors;

  for(int i = 0; i < NUM_EIGEN_FACES; i++)
  {
      Mat eigenFace = eigenVectors.row(i).reshape(3,sz.height);
      eigenFaces.push_back(eigenFace);
  }

  Mat output;
  resize(averageFace, output, Size(), 2, 2);

  namedWindow("Result", WINDOW_AUTOSIZE);
  imshow("Result", output);

  namedWindow("Trackbars", WINDOW_AUTOSIZE);
  for(int i = 0; i < NUM_EIGEN_FACES; i++)
  {
    sliderValues[i] = MAX_SLIDER_VALUE/2;
    createTrackbar( "Weight" + to_string(i), "Trackbars", &sliderValues[i], MAX_SLIDER_VALUE, createNewFace);
  }


  setMouseCallback("Result", resetSliderValues);

  cout << "Usage:" << endl
  << "\tChange the weights using the sliders" << endl
  << "\tClick on the result window to reset sliders" << endl
  << "\tHit ESC to terminate program." << endl;

  waitKey(0);
  destroyAllWindows();

  return 0;
}

