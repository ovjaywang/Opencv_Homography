#pragma once

#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/nonfree/nonfree.hpp>  
#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <cv.h>
#include "cxcore.h"
#include <highgui.h>
#include "Findhomography.h"
#include "HomographyAjustment.h"
using namespace cv;
class Stiching
{
public:
	Stiching();
	~Stiching();
public:
	vector<double> Parameters;
	HomographyAjustment  m_adjustment;
	Findhomography m_Findhomography ;
	int ProcessStitching(String strs[], double wid, double up_hei, double down_hei);
};
