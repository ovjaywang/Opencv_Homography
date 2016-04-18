#pragma once
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/nonfree/nonfree.hpp>  
#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <cv.h>
#include "cxcore.h"
#include <highgui.h>
#include <iostream>  
#include <stdio.h>
#include <fstream>
#include <iomanip>
using namespace cv;
class Homography
{
public:
	Homography();
	~Homography();

public:
	CvMat* lsq_homog(vector<Point2f> leftPts, vector<Point2f> RightPts, int n);
};

