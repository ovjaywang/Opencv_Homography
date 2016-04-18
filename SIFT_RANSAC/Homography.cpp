#include "Homography.h"


Homography::Homography()
{
}


Homography::~Homography()
{
}

CvMat* lsq_homog(vector<Point2f> leftPts, vector<Point2f> RightPts, int n)
{
	CvMat* H, *A, *B, X;
	double x[9];//数组x中的元素就是变换矩阵H中的值
	int i;

	//输入点对个数不够4
	if (n < 4)
	{
		fprintf(stderr, "Warning: too few points in lsq_homog(), %s line %d\n",
			__FILE__, __LINE__);
		return NULL;
	}

	//将变换矩阵H展开到一个8维列向量X中，使得AX=B，这样只需一次解线性方程组即可求出X，然后再根据X恢复H
	/* set up matrices so we can unstack homography into X; AX = B */
	A = cvCreateMat(2 * n, 8, CV_64FC1);//创建2n*8的矩阵，一般是8*8
	B = cvCreateMat(2 * n, 1, CV_64FC1);//创建2n*1的矩阵，一般是8*1
	X = cvMat(8, 1, CV_64FC1, x);//创建8*1的矩阵，指定数据为x
	H = cvCreateMat(3, 3, CV_64FC1);//创建3*3的矩阵
	cvZero(A);//将A清零

	//由于是展开计算，需要根据原来的矩阵计算法则重新分配矩阵A和B的值的排列
	for (i = 0; i < n; i++)
	{
		cvmSet(A, i, 0, leftPts[i].x);//设置矩阵A的i行0列的值为pts[i].x
		cvmSet(A, i + n, 3, leftPts[i].x);
		cvmSet(A, i, 1, leftPts[i].y);
		cvmSet(A, i + n, 4, leftPts[i].y);
		cvmSet(A, i, 2, 1.0);
		cvmSet(A, i + n, 5, 1.0);
		cvmSet(A, i, 6, -leftPts[i].x * RightPts[i].x);
		cvmSet(A, i, 7, -leftPts[i].y * RightPts[i].x);
		cvmSet(A, i + n, 6, -leftPts[i].x * RightPts[i].y);
		cvmSet(A, i + n, 7, -leftPts[i].y * RightPts[i].y);
		cvmSet(B, i, 0, RightPts[i].x);
		cvmSet(B, i + n, 0, RightPts[i].y);
	}

	//调用OpenCV函数，解线性方程组
	cvSolve(A, B, &X, CV_SVD);//求X，使得AX=B
	x[8] = 1.0;//变换矩阵的[3][3]位置的值为固定值1
	X = cvMat(3, 3, CV_64FC1, x);
	cvConvert(&X, H);//将数组转换为矩阵

	cvReleaseMat(&A);
	cvReleaseMat(&B);
	return H;
}