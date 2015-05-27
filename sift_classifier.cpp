// BoFSIFT.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\nonfree\nonfree.hpp>
#include <C:\Users\USER\Desktop\CaptureVideoFromFile\Debug\wait.h>
#include "opencv\ml.h"
#include <stdio.h>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <opencv2/legacy/legacy.hpp>
#include "opencv.hpp"
#include <conio.h>

using namespace cv;
using namespace std;

#define DICTIONARY_BUILD 0 // set DICTIONARY_BUILD 1 to do Step 1, otherwise it goes to step 2




int main()
{	
#if DICTIONARY_BUILD == 0
	
	//Step 1 - Obtain the set of bags of features.

	//to store the input file names
	char * filename = new char[100];		
	//to store the current input image
	Mat input;	

	//To store the keypoints that will be extracted by SIFT
	vector<KeyPoint> keypoints;
	//To store the SIFT descriptor of current image
	Mat descriptor;
	//To store all the descriptors that are extracted from all the images
	Mat featuresUnclustered;
	//The SIFT feature extractor and descriptor
	SiftDescriptorExtractor detector;	

	/*
	cv::Ptr<cv::DescriptorMatcher> matcher =    cv::DescriptorMatcher::create("FlannBased");
    cv::Ptr<cv::DescriptorExtractor> extractor = new cv::SurfDescriptorExtractor();
    cv::BOWImgDescriptorExtractor dextract( extractor, matcher );
    cv::SurfFeatureDetector detector(500);
	*/
	
	int i,j;
	float kl=0,l=0;
	for(j=1;j<=3;j++)
	for(i=1;i<=3;i++){
		sprintf(filename,"%d%s%d%s",j," (",i,").jpg");
		//create the file name of an image
		//open the file
		input = imread(filename, CV_LOAD_IMAGE_GRAYSCALE); //Load as grayscale				
		//detect feature points
		detector.detect(input, keypoints);
		//compute the descriptors for each keypoint
		detector.compute(input, keypoints,descriptor);		
		//put the all feature descriptors in a single Mat object 
		featuresUnclustered.push_back(descriptor);		
		//print the percentage
		l++;
		kl=(l*100)/9;
		cout<<kl<<"% done\n";
			
	}	

	int dictionarySize=100;
	//define Term Criteria
	TermCriteria tc(CV_TERMCRIT_ITER,100,0.001);
	//retries number
	int retries=1;
	//necessary flags
	int flags=KMEANS_PP_CENTERS;
	//Create the BoW (or BoF) trainer
	BOWKMeansTrainer bowTrainer(dictionarySize,tc,retries,flags);
	//cluster the feature vectors
	Mat dictionary=bowTrainer.cluster(featuresUnclustered);	
	//store the vocabulary
	FileStorage fs("dictionary1.yml", FileStorage::WRITE);
	fs << "vocabulary" << dictionary;
	fs.release();
	cout<<"Saving BoW dictionary\n";

    
	//create a nearest neighbor matcher
	Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
	//create Sift feature point extracter
	Ptr<FeatureDetector> detector1(new SiftFeatureDetector());
	//create Sift descriptor extractor
	Ptr<DescriptorExtractor> extractor(new SiftDescriptorExtractor);	
	//create BoF (or BoW) descriptor extractor
	BOWImgDescriptorExtractor bowDE(extractor,matcher);
	//Set the dictionary with the vocabulary we created in the first step
	bowDE.setVocabulary(dictionary);

	cout<<"extracting histograms in the form of BOW for each image "<<endl;
	Mat labels(0, 1, CV_32FC1);
	Mat trainingData(0, dictionarySize, CV_32FC1);
	int k=0;
	vector<KeyPoint> keypoint1;
	Mat bowDescriptor1;
	Mat img2;
	//extracting histogram in the form of bow for each image 
for(j=1;j<=3;j++)
	for(i=1;i<=3;i++){
				
				
					sprintf( filename,"%d%s%d%s",j," (",i,").jpg");
					img2 = cvLoadImage(filename,0);
				
					detector.detect(img2, keypoint1);
				
						
						bowDE.compute(img2, keypoint1, bowDescriptor1);
						
						trainingData.push_back(bowDescriptor1);
						
						labels.push_back((float) j);
	}
	cout<<"Done!\n";


	CvSVMParams params;
	params.kernel_type=CvSVM::RBF;
	params.svm_type=CvSVM::C_SVC;
	params.gamma=0.50625000000000009;
	params.C=312.50000000000000;
	params.term_crit=cvTermCriteria(CV_TERMCRIT_ITER,100,0.000001);
	CvSVM svm;



	printf("%s\n","Training SVM classifier");

	bool res=svm.train(trainingData,labels,cv::Mat(),cv::Mat(),params);

	svm.save("svm-classifier1.xml");

	printf("%s\n","Done!, saved as an external file");


	cout<<"Performing image classification\n\n Input image:"<<endl;

	vector<KeyPoint> keypoint2;
	Mat bowDescriptor2;

	char ch[5];
	Mat results(0, 1, CV_32FC1);
	j=2;
				
					sprintf( filename,"%s%d%s","q",j,".jpg");
					img2 = cvLoadImage(filename,0);
					namedWindow("Image");
					imshow("Image",img2);
					cvWaitKey(10);
					Mat img_keypoints_2;
				
					detector.detect(img2, keypoint2);
						bowDE.compute(img2, keypoint2, bowDescriptor2);

						drawKeypoints(img2,keypoint2,img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

						namedWindow("Image-detected");
						imshow("Image-detected",img_keypoints_2);
						cvWaitKey(10);
											

						int it = svm.predict(bowDescriptor2);
						results.push_back((float)it);
						std::cout<<"\n!The image belongs to class:!\n";
						(it == 1) ? cout<<"Airplane\n" : 
						(it == 2) ? cout<<"Car\n" :
						(it == 3) ? cout<<"Cheetah\n" : cout<<"Oh oh!\n";

	_getch();
	cvDestroyAllWindows();
	return 0;
	
#else
	//Step 2 - Obtain the BoF descriptor for given image/video frame. 

    //prepare BOW descriptor extractor from the dictionary    
	Mat dictionary; 
	FileStorage fs("dictionary1.yml", FileStorage::READ);
	fs["vocabulary"] >> dictionary;
	fs.release();	
    
	//create a nearest neighbor matcher
	Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
	//create Sift feature point extracter
	Ptr<FeatureDetector> detector(new SiftFeatureDetector());
	//create Sift descriptor extractor
	Ptr<DescriptorExtractor> extractor(new SiftDescriptorExtractor);	
	//create BoF (or BoW) descriptor extractor
	BOWImgDescriptorExtractor bowDE(extractor,matcher);
	//Set the dictionary with the vocabulary we created in the first step
	bowDE.setVocabulary(dictionary);

	//To store the image file name
	char * filename = new char[100];
	//To store the image tag name - only for save the descriptor in a file
	char * imageTag = new char[10];

	Mat bowDescriptor; 
	FileStorage fs1("descriptor.yml", FileStorage::READ);
	fs1["img1"] >> bowDescriptor;
	
	vector<KeyPoint> keypoints;
	vector<KeyPoint> keypoints1;

	//open the file to write the resultant descriptor
	FileStorage fs1("descriptor.yml", FileStorage::WRITE);	
	
	//the image file with the location. change it according to your image file location
	sprintf(filename,"in_final.jpg");		
	//read the image
	Mat img=imread(filename,CV_LOAD_IMAGE_GRAYSCALE);		
	//To store the keypoints that will be extracted by SIFT	
	//Detect SIFT keypoints (or feature points)
	detector->detect(img,keypoints);
	//To store the BoW (or BoF) representation of the image
	//Mat bowDescriptor;		
	//extract BoW (or BoF) descriptor from given image
	bowDE.compute(img,keypoints,bowDescriptor);
	

	//********** 2nd image****
	sprintf(filename,"2.jpg");		
	//read the image
	Mat img1=imread(filename,CV_LOAD_IMAGE_GRAYSCALE);		
	//To store the keypoints that will be extracted by SIFT
	//vector<KeyPoint> keypoints;		
	//Detect SIFT keypoints (or feature points)
	detector->detect(img1,keypoints1);
	//To store the BoW (or BoF) representation of the image
	Mat bowDescriptor1;		
	//extract BoW (or BoF) descriptor from given image
	bowDE.compute(img1,keypoints1,bowDescriptor1);

	//*******************************************

	//**** Matcher Analysis*****
	FlannBasedMatcher matcher1;
  std::vector<DMatch> matches;
  matcher1.match(bowDescriptor, bowDescriptor1, matches);

  double max_dist = 0; double min_dist = 100;

  //-- Quick calculation of max and min distances between keypoints
  for (size_t i = 0; i < (size_t)bowDescriptor.rows; i++ ) {
    double dist = matches[i].distance;
    if (dist < min_dist) min_dist = dist;
    if (dist > max_dist) max_dist = dist;
  }

  //std::cout<<"max dist ="<<max_dist<<" min dist ="<<min_dist<<"\n";

 std::vector<DMatch> good_matches;
double good_matches_sum = 0.0;
	for (size_t i = 0; i < (size_t)bowDescriptor.rows; i++ ) {
  if( matches[i].distance <= max(2*min_dist, 0.02) ) {
    good_matches.push_back(matches[i]);
    good_matches_sum += matches[i].distance;
  }
}	
	double score = (double)good_matches_sum / (double)good_matches.size();

if (score < 0.18) {
  std::cout<<"Matched Score"<<score;
} else {
  std::cout<<"Not Matched Score"<<score;
}
	wait(10);	


Mat img_matches;
  drawMatches( img, keypoints, img1, keypoints1,
               good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
	imshow("image", img_matches);
	cvWaitKey(100);
	 
	

	//prepare the yml (some what similar to xml) file
	sprintf(imageTag,"img1");			
	//write the new BoF descriptor to the file
	fs1 << imageTag << bowDescriptor;		

	//You may use this descriptor for classifying the image.
			
	//release the file storage
	fs1.release(); 
#endif
	_getch();
	printf("\ndone\n");	
    return 0;
}