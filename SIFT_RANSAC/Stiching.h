#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/nonfree/nonfree.hpp>  
#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <cv.h>
#include "cxcore.h"
#include <highgui.h>
#include "Findhomography.h"
class Stiching
{
public:
	Stiching();
	~Stiching();
	Findhomography m_Findhomography;
};
