#pragma once
#include <stdlib.h> 
#include <stdio.h> 
#include <math.h> 
#include <highgui.h> 
using namespace cv;
class Findhomography
{
public:
	Findhomography();
	~Findhomography();
public:
	void RANSAC_homography(int num, CvPoint2D64f *m1, CvPoint2D64f *m2, CvMat *H, CvMat *inlier_mask);
	int dostiching(String sstr[]);
	void Normalization(int num, CvPoint2D64f *p, CvMat *T);
	int ComputeNumberOfInliers(int num, CvPoint2D64f *p1, CvPoint2D64f *p2, CvMat *H,
		CvMat *inlier_mask, double*dist_std);
	void ComputeH(int n, CvPoint2D64f *p1, CvPoint2D64f *p2, CvMat *H);
	void ComputeH0(int n, CvPoint2D64f *p1, CvPoint2D64f *p2, CvMat *H);
	bool isColinear(int num, CvPoint2D64f *p);
	int CornerPointMatching_NCC(IplImage *img1, IplImage *img2, CvPoint *p1, int num1, CvPoint *p2, int num2, CvPoint2D64f *m1, CvPoint2D64f *m2);
	int DetectCorner(IplImage *img, CvPoint *corner);
	int Corner_Uniqueness(CvPoint *corner, int num, double*corner_cost, CvPoint	curr_point, double curr_cost);
	void Gradient_Sobel(IplImage *img, CvMat* I_x, CvMat* I_y);
};

