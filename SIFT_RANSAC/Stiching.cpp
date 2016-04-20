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
#include "Stiching.h"
using namespace cv; 
#define MAX_POINT_SIZE 20000
#define MAX_CORNERPOINT_NUM 10000
	//RANSAC_homography(num_matched, matched1, matched2, H, inlier_mask);
Stiching::Stiching(){
	
}
Stiching::~Stiching(){

}
int Stiching::ProcessStitching(String strs[])
{
	ofstream ofs;
	ofstream ofs_inline;
#ifdef _DEBUG
	ofs.open("E:\\feature_matched.txt", ios_base::trunc | ios_base::in);
	ofs_inline.open("E:\\inline_matched.txt", ios_base::trunc | ios_base::in);
	ofs << "    X左            Y左             X右           Y右" << endl;
#endif // _DEBUG
	initModule_nonfree();//初始化模块，使用SIFT或SURF时用到 
	Ptr<FeatureDetector> detector = FeatureDetector::create( "SURF" );
	//创建SIFT特征检测器，可改成SURF/ORB SURF稳定 精度和结果与SIFT类似 但效率高出不少
	Ptr<DescriptorExtractor> descriptor_extractor = DescriptorExtractor::create( "SURF" );//创建特征向量生成器，可改成SURF/ORB
	Ptr<DescriptorMatcher> descriptor_matcher = DescriptorMatcher::create( "BruteForce" );//创建特征匹配器  
//	Ptr<DescriptorMatcher> descriptor_matcher = DescriptorMatcher::create("FlannBased");//创建特征匹配器  
	if( detector.empty() || descriptor_extractor.empty() )  
		cout<<"fail to create detector!";  

	//读入图像  
	Mat img1 = imread(strs[0]);
	Mat img2 = imread(strs[1]);
	//针对图像进行ROI选取
	Rect roi(2888, 0, 1400, 1600);
	Rect roi2(0, 0, 1400, 1600);
	Mat roi_of_image1 = img1(roi);
	Mat roi_of_image2 = img2(roi2);
	//特征点检测  
	double t = getTickCount();//当前滴答数  
	vector<KeyPoint> m_LeftKey,m_RightKey;
	detector->detect(roi_of_image1, m_LeftKey);//检测img1中的SIFT特征点，存储到m_LeftKey中  
	detector->detect(roi_of_image2, m_RightKey);
	cout<<"图像1特征点个数:"<<m_LeftKey.size()<<"\t\t图像2特征点个数:"<<m_RightKey.size()<<endl;  

	//根据特征点计算特征描述子矩阵，即特征向量矩阵  
	Mat descriptors1,descriptors2;  
	descriptor_extractor->compute(roi_of_image1, m_LeftKey, descriptors1);
	descriptor_extractor->compute(roi_of_image2, m_RightKey, descriptors2);
	double t1 = ((double)getTickCount() - t) / getTickFrequency();
	cout<<"SURF算法用时："<<t1<<"秒"<<endl;  
	//ROI提取的特征点以子图为坐标系 因此大图需要转换
	for (int i = 0; i<m_LeftKey.size(); i++)
	{
		m_LeftKey[i].pt.x += 2888;//左点
	}
	//画出特征点  
	Mat img_m_LeftKey,img_m_RightKey;  
	drawKeypoints(img1,m_LeftKey,img_m_LeftKey,Scalar::all(-1),0);  
	drawKeypoints(img2,m_RightKey,img_m_RightKey,Scalar::all(-1),0);  
	//imshow("Src1",img_m_LeftKey);  
	//imshow("Src2",img_m_RightKey);  
	cvSaveImage("E:\\Src1.jpg", &IplImage(img_m_LeftKey));
	cvSaveImage("E:\\Src2.jpg", &IplImage(img_m_RightKey));
	//特征匹配  
	vector<DMatch> matches;//匹配结果
	//①第一次筛选 opencv内部的matcher匹配器 应该使用kdtree进行向量欧几里得距离的相似匹配
	descriptor_matcher->match( descriptors1, descriptors2, matches );//匹配两个图像的特征矩阵  
	cout<<"Match个数："<<matches.size()<<endl;  

	//计算匹配结果中距离的最大和最小值  
	//距离是指两个特征向量间的欧式距离，表明两个特征的差异，值越小表明两个特征点越接近  
	double max_dist = 0;  
	double min_dist = 100;
	for(int i=0; i<matches.size(); i++)  
	{  
		double dist = matches[i].distance;  
		if(dist < min_dist) min_dist = dist;  
		if(dist > max_dist) max_dist = dist;  
	}  
	cout<<"最大距离："<<max_dist<<endl;  
	cout<<"最小距离："<<min_dist<<endl;  

	//筛选出较好的匹配点  
	vector<DMatch> goodMatches;
	//②第二次筛选 根据上一步初步得到的匹配点 计算欧几里得距离距离 与最大/最小作对比
	for(int i=0; i<matches.size(); i++)  
	{  
		if(matches[i].distance < 0.2 * max_dist)  
			//这里 只选取最大值五分之一的 可以调整得更小以筛选更接近的点集 但匹配点会相对减少
		//if (matches[i].distance < 3 * min_dist)
		{  
			goodMatches.push_back(matches[i]);  
		}  
	}  
	cout<<"goodMatch个数："<<goodMatches.size()<<endl;  

	//画出匹配结果  
	Mat img_matches;  //显示匹配的图像
	//红色连接的是匹配的特征点对，绿色是未匹配的特征点  
	drawMatches(img1,m_LeftKey,img2,m_RightKey,goodMatches,img_matches,  
		Scalar::all(-1)/*CV_RGB(255,0,0)*/,CV_RGB(0,255,0),Mat(),2);  
	double ttt = ((double)getTickCount() - t) / getTickFrequency();
	cout << "截止粗匹配用时：" << ttt << "秒" << endl;
//	imshow("MatchSIFT",img_matches);
	cvSaveImage("E:\\MatchSIFT.jpg", &IplImage(img_matches));
	IplImage result=img_matches;

	//waitKey(0);  //等待输入esc键退出


	//RANSAC匹配过程
	vector<DMatch> m_Matches=goodMatches;
	// 分配空间
	int ptCount = (int)m_Matches.size();
	Mat p1(ptCount, 2, CV_32F);
	Mat p2(ptCount, 2, CV_32F);

	// 把Keypoint转换为Mat  
	Point2f pt;
	for (int i=0; i<ptCount; i++)
	{
		pt = m_LeftKey[m_Matches[i].queryIdx].pt;//左点
		p1.at<float>(i, 0) = pt.x;
		p1.at<float>(i, 1) = pt.y;

		pt = m_RightKey[m_Matches[i].trainIdx].pt;//右点
		p2.at<float>(i, 0) = pt.x;
		p2.at<float>(i, 1) = pt.y;
	}

	// ③第三次筛选 用RANSAC方法计算F 根据F矩阵计算得到的极限筛选最靠谱的点
	Mat m_Fundamental;
	vector<uchar> m_RANSACStatus;       //这个变量用于存储RANSAC后每个点的状态
	findFundamentalMat(p1, p2, m_RANSACStatus, FM_RANSAC);
	// 计算野点个数
	double tt = ((double)getTickCount() - t) / getTickFrequency();
	cout << "截止Ransac匹配用时：" << tt << "秒" << endl;
	int OutlinerCount = 0;
	for (int i=0; i<ptCount; i++)
	{
		if (m_RANSACStatus[i] == 0)    // 状态为0表示野点
		{
			OutlinerCount++;
		}
	}
	int InlinerCount = ptCount - OutlinerCount;   // 计算内点
	cout<<"总点数为 ： "<<m_RANSACStatus.size()<<"  内点数为："<<InlinerCount<<endl;

	
   // 这三个变量用于保存内点和匹配关系
   vector<Point2f> m_LeftInlier;//左内点
   vector<Point2f> m_RightInlier;//右内点
   vector<DMatch> m_InlierMatches;//匹配对

	m_InlierMatches.resize(InlinerCount);
	m_LeftInlier.resize(InlinerCount);
	m_RightInlier.resize(InlinerCount);
	InlinerCount=0;
	float inlier_minRx=img1.cols;        //用于存储内点中右图最小横坐标，以便后续融合 裁剪 默认图像为横向
	
	for (int i=0; i<ptCount; i++)
	{
		if (m_RANSACStatus[i] != 0)
		{
			m_LeftInlier[InlinerCount].x = p1.at<float>(i, 0);
			m_LeftInlier[InlinerCount].y = p1.at<float>(i, 1);
			m_RightInlier[InlinerCount].x = p2.at<float>(i, 0);
			m_RightInlier[InlinerCount].y = p2.at<float>(i, 1);
			m_InlierMatches[InlinerCount].queryIdx = InlinerCount;
			m_InlierMatches[InlinerCount].trainIdx = InlinerCount;
			if(m_RightInlier[InlinerCount].x<inlier_minRx) inlier_minRx=m_RightInlier[InlinerCount].x;   //存储内点中右图最小横坐标 为了找出图像最左侧	
			InlinerCount++;
		}
	}

	// 把内点转换为drawMatches可以使用的格式
	vector<KeyPoint> key1(InlinerCount);
	vector<KeyPoint> key2(InlinerCount);
	KeyPoint::convert(m_LeftInlier, key1);
	KeyPoint::convert(m_RightInlier, key2);

	//CvMat* HH, *A, *B, X;
	//double x[9];//数组x中的元素就是变换矩阵H中的值
	//int i;

	////输入点对个数不够4
	////将变换矩阵H展开到一个8维列向量X中，使得AX=B，这样只需一次解线性方程组即可求出X，然后再根据X恢复H
	///* set up matrices so we can unstack homography into X; AX = B */
	//A = cvCreateMat(2 * key1.size(), 8, CV_64FC1);//创建2n*8的矩阵，一般是8*8
	//B = cvCreateMat(2 * key1.size(), 1, CV_64FC1);//创建2n*1的矩阵，一般是8*1
	//X = cvMat(8, 1, CV_64FC1, x);//创建8*1的矩阵，指定数据为x
	//HH = cvCreateMat(3, 3, CV_64FC1);//创建3*3的矩阵
	//cvZero(A);//将A清零

	////由于是展开计算，需要根据原来的矩阵计算法则重新分配矩阵A和B的值的排列
	//for (int ii = 0; ii < key1.size(); ii++)
	//{
	//	//cvmSet(A, i, 0, key1[i].pt.x);//设置矩阵A的i行0列的值为pts[i].x
	//	//cvmSet(A, i + key1.size(), 3, key1[i].pt.x);
	//	//cvmSet(A, i, 1, key1[i].pt.y);
	//	//cvmSet(A, i + key1.size(), 4, key1[i].pt.y);
	//	//cvmSet(A, i, 2, 1.0);
	//	//cvmSet(A, i + key1.size(), 5, 1.0);
	//	//cvmSet(A, i, 6, -key1[i].pt.x *  key2[i].pt.x);
	//	//cvmSet(A, i, 7, -key1[i].pt.y *  key2[i].pt.x);
	//	//cvmSet(A, i + key1.size(), 6, -key1[i].pt.x *  key2[i].pt.y);
	//	//cvmSet(A, i + key1.size(), 7, -key1[i].pt.y *  key2[i].pt.y);
	//	//cvmSet(B, i, 0, key2[i].pt.x);
	//	//cvmSet(B, i + key1.size(), 0, key2[i].pt.y);

	//	cvmSet(A, 2 * ii, 0, key1[ii].pt.x);//设置矩阵A的i行0列的值为pts[i].x
	//	cvmSet(A, 2 * ii + 1, 3, key1[ii].pt.x);
	//	cvmSet(A, 2 * ii, 1, key1[ii].pt.y);
	//	cvmSet(A, 2 * ii + 1, 4, key1[ii].pt.y);
	//	cvmSet(A, 2 * ii, 2, 1.0);
	//	cvmSet(A, 2 * ii + 1, 5, 1.0);
	//	cvmSet(A, 2 * ii, 6, -key1[ii].pt.x *  key2[ii].pt.x);
	//	cvmSet(A, 2 * ii, 7, -key1[ii].pt.y *  key2[ii].pt.x);
	//	cvmSet(A, 2 * ii + 1, 6, -key1[ii].pt.x *  key2[ii].pt.y);
	//	cvmSet(A, 2 * ii + 1, 7, -key1[ii].pt.y *  key2[ii].pt.y);

	//	cvmSet(B, 2 * ii, 0, key2[ii].pt.x);
	//	cvmSet(B, 2 * ii + 1, 0, key2[ii].pt.y);
	//}

	////调用OpenCV函数，解线性方程组
	//cvSolve(A, B, &X, CV_SVD);//求X，使得AX=B
	//x[8] = 1.0;//变换矩阵的[3][3]位置的值为固定值1
	//X = cvMat(3, 3, CV_64FC1, x);
	//cvConvert(&X, HH);//将数组转换为矩阵

	//cvReleaseMat(&A);
	//cvReleaseMat(&B);
	//cout <<endl<< "--SVM算法计算结果为--   " << endl;
	//for (int row = 0; row < (*HH).rows; row++)
	//{
	//	float* pptr = (float*)((*HH).data.ptr + row * (*HH).step);//第row行数据的起始指针
	//	for (int col = 0; col < (*HH).cols; col++)
	//	{
	//		//cout << *(pptr + 1 * col) <<"  ";
	//		cout << cvmGet(HH, row, col) << "\t";
	//	}
	//	cout << endl;
	//}
	//结束homography
	CvPoint2D64f matched1[MAX_CORNERPOINT_NUM];
	CvPoint2D64f matched2[MAX_CORNERPOINT_NUM];
	//解算H之前输出配对点
	for (int i = 0; i < key1.size(); i++){		
#ifdef _DEBUG
		ofs << key1[i].pt.x << "   " << key1[i].pt.y << "   " << key2[i].pt.x << "   " << key2[i].pt.y << endl;
		matched1[i].x = key1[i].pt.x;
		matched1[i].y = key1[i].pt.y;
		matched2[i].x = key2[i].pt.x;
		matched2[i].y = key2[i].pt.y;
#endif // _DEBUG
	}
	CvMat *HH = cvCreateMat(3, 3, CV_64FC1);
	CvMat *inlier_mask = cvCreateMat(key1.size(), 1, CV_64FC1);
	
	Findhomography homo;
	homo.RANSAC_homography(key1.size(), matched1, matched2, HH, inlier_mask);
	CvPoint newmatched;
	int num_inlier = 0;
	for (int i = 0; i<key1.size(); i++){
		if (cvmGet(inlier_mask, i, 0) == 1){
			ofs_inline << matched1[i].x << "\t" << matched1[i].y << "\t" << matched2[i].x << "\t" << matched2[i].y << endl;
			num_inlier++;
		}
	}
	printf("内点个数为 : %d\n", num_inlier);



	cout <<endl<< "--直接Ransac算法计算结果为--   " << endl;
	double c33 = cvmGet(HH, 2, 2);
	for (int row = 0; row < (*HH).rows; row++)
	{
		//float* pptr = (float*)((*HH).data.ptr + row * (*HH).step);//第row行数据的起始指针
		for (int col = 0; col < (*HH).cols; col++)
		{
			//cout << *(pptr + 1 * col) <<"  ";
			cout << cvmGet(HH, row, col)/c33 << "\t";
		}
		cout << endl;
	}
    // 显示计算F过后的内点匹配
	Mat OutImage;
	drawMatches(img1, key1, img2, key2, m_InlierMatches, OutImage);
//	cvNamedWindow( "Match features", 1);
//	cvShowImage("Match features", &IplImage(OutImage));
	cvSaveImage("Match features.jpg", &IplImage(OutImage));
	//waitKey(0);

	//cvDestroyAllWindows();

	//矩阵H用以存储RANSAC得到的单应矩阵
	//Mat H = findHomography( m_LeftInlier, m_RightInlier, RANSAC );
//	Mat H=Mat::Mat(HH, true);


	//④第四次筛选 在解算H时一边Ransac一边选择最大一只鸡
	Mat H = findHomography(m_LeftInlier, m_RightInlier, RANSAC);
	cout<< endl<< H << endl;

	//double aa1 = 1.045972;
	//double aa2 = -0.000410;
	//double aa3 = -3101.372651;
	//double bb1 = 0.004664;
	//double bb2 = 1.032072;
	//double bb3 = 16.766268;
	//double cc1 = 0.000010;
	//double cc2 = -0.000005;
	//double cc3 = 1;
	double aa1 = 1.027131;
	double aa2 = 0.007651;
	double aa3 = -2985.513937;
	double bb1 = 0.002849;
	double bb2 = 1.017038;
	double bb3 = 18.118161;
	double cc1 = 0.000005;
	double cc2 = 0.000005;
	double cc3 = 1;

	H.at<double>(0, 0) = aa1 / cc3;
	H.at<double>(0, 1) = aa2 / cc3;
	H.at<double>(0, 2) = aa3/ cc3;
	H.at<double>(1, 0) = bb1 / cc3;
	H.at<double>(1, 1) = bb2 / cc3;
	H.at<double>(1, 2) = bb3 / cc3;
	H.at<double>(2, 0) = cc1 / cc3;
	H.at<double>(2, 1) = cc2 / cc3;
	H.at<double>(2, 2) = 1;

	cout <<endl<< H << endl;
	//存储左图四角，及其变换到右图位置
	std::vector<Point2f> obj_corners(4);
	obj_corners[0] = Point(0,0); obj_corners[1] = Point( img1.cols, 0 );
	obj_corners[2] = Point( img1.cols, img1.rows ); obj_corners[3] = Point( 0, img1.rows );
	std::vector<Point2f> scene_corners(4);
	perspectiveTransform( obj_corners, scene_corners, H);

	//画出左图经过H变换后在图像中位置
	Point2f offset( (float)img1.cols, 0);//左图右下角 作为平移向量  
	line( OutImage, scene_corners[0]+offset, scene_corners[1]+offset, Scalar( 0, 255, 0), 4 );
	line( OutImage, scene_corners[1]+offset, scene_corners[2]+offset, Scalar( 0, 255, 0), 4 );
	line( OutImage, scene_corners[2]+offset, scene_corners[3]+offset, Scalar( 0, 255, 0), 4 );
	line( OutImage, scene_corners[3]+offset, scene_corners[0]+offset, Scalar( 0, 255, 0), 4 );
//	imshow( "Good Matches & Object detection", OutImage );
	cvSaveImage("E:\\Good Matches Object detection.jpg", &IplImage(OutImage));
/*		while(1)
	{
		if(waitKey(100)==19) cvSaveImage("E:\\Good Matches Object detection.jpg",  &IplImage(OutImage));  
		if(waitKey(100)==27) break;
	}      */                                                        //按esc继续，ctl+s保存图像
	
	int drift = scene_corners[1].x;                                                        //储存偏移量

	//新建一个矩阵存储配准后整幅图四角的位置
	int width = int(max(abs(scene_corners[1].x), abs(scene_corners[2].x)));
	int height= img1.rows;                                                                  //或者：int height = int(max(abs(scene_corners[2].y), abs(scene_corners[3].y)));
	float origin_x=0,origin_y=0;
		if(scene_corners[0].x<0) {//如果映射后 左图左上角偏移到坐标x=0左边
			if (scene_corners[3].x<0) origin_x+=min(scene_corners[0].x,scene_corners[3].x);//又如果左下角也跑到x=0左边 设定整幅图的左边界
		else origin_x+=scene_corners[0].x;}//如果左下角没有跑偏 则左边界赋值
		width-=int(origin_x);
	if(scene_corners[0].y<0) {//同理 设定上边界
		if (scene_corners[1].y) origin_y+=min(scene_corners[0].y,scene_corners[1].y);
		else origin_y+=scene_corners[0].y;}
	//可选：height-=int(origin_y);
	Mat imageturn=Mat::zeros(width,height,img1.type());//横拼 高度不变

	//获取新的变换矩阵，使图像完整显示
	for (int i=0;i<4;i++) {scene_corners[i].x -= origin_x; } 	//横拼  可选纵拼：scene_corners[i].y -= (float)origin_y; }
	Mat H1=getPerspectiveTransform(obj_corners, scene_corners);
	cout <<endl<< H1 << endl;
	//对左图进行图像变换，显示效果
	warpPerspective(img1,imageturn,H1,Size(width,height));	
//	imshow("image_Perspective", imageturn);
	cvSaveImage("E:\\Left image after perspectiving.jpg", &IplImage(imageturn));
//	waitKey(0);

	//图像融合
	int width_ol=width-int(inlier_minRx-origin_x);
	int start_x=int(inlier_minRx-origin_x);
    cout<<"width: "<<width<<"\timg1.width: "<<img1.cols<<"\tstart_x: "<<start_x<<"\twidth_ol: "<<width_ol<<endl;
	uchar* ptr=imageturn.data;//使用指针给数据赋值
	double alpha=0, beta=1;
	for (int row=0;row<height;row++) {
		ptr=imageturn.data+row*imageturn.step+(start_x)*imageturn.elemSize();//当前行指针
		for(int col=0;col<width_ol;col++)
		{
			uchar* ptr_c1=ptr+imageturn.elemSize1();
			uchar* ptr_c2=ptr_c1+imageturn.elemSize1();
			uchar* ptr2=img2.data+row*img2.step+(col+int(inlier_minRx))*img2.elemSize();
			uchar* ptr2_c1=ptr2+img2.elemSize1();  
			uchar* ptr2_c2=ptr2_c1+img2.elemSize1();

			alpha=double(col)/double(width_ol); beta=1-alpha;//图像融合 越靠近拼接线的地方 使用平均值 往左往右分别使用左右图更多比例

			if (*ptr==0&&*ptr_c1==0&&*ptr_c2==0) {
				*ptr=(*ptr2);
				*ptr_c1=(*ptr2_c1);
				*ptr_c2=(*ptr2_c2);
			}

			*ptr=(*ptr)*beta+(*ptr2)*alpha;
			*ptr_c1=(*ptr_c1)*beta+(*ptr2_c1)*alpha;
			*ptr_c2=(*ptr_c2)*beta+(*ptr2_c2)*alpha;

			ptr+=imageturn.elemSize();
		}	}
	
//	imshow("image_overlap", imageturn);
	cvSaveImage("E:\\Left image with overlap.jpg", &IplImage(imageturn));
	//waitKey(0);

	Mat img_result=Mat::zeros(height,width+img2.cols-drift,img1.type());
	uchar* ptr_r=imageturn.data;
	
	for (int row=0;row<height;row++) {
		ptr_r=img_result.data+row*img_result.step;

		for(int col=0;col<imageturn.cols;col++)
		{
			uchar* ptr_rc1=ptr_r+imageturn.elemSize1();  uchar*  ptr_rc2=ptr_rc1+imageturn.elemSize1();

			uchar* ptr=imageturn.data+row*imageturn.step+col*imageturn.elemSize();
			uchar* ptr_c1=ptr+imageturn.elemSize1();  uchar*  ptr_c2=ptr_c1+imageturn.elemSize1();

			*ptr_r=*ptr;
			*ptr_rc1=*ptr_c1;
			*ptr_rc2=*ptr_c2;

			ptr_r+=img_result.elemSize();
		}	

		ptr_r=img_result.data+row*img_result.step+imageturn.cols*img_result.elemSize();
		for(int col=imageturn.cols;col<img_result.cols;col++)
		{
			uchar* ptr_rc1=ptr_r+imageturn.elemSize1();  uchar*  ptr_rc2=ptr_rc1+imageturn.elemSize1();

			uchar* ptr2=img2.data+row*img2.step+(col-imageturn.cols+drift)*img2.elemSize();
			uchar* ptr2_c1=ptr2+img2.elemSize1();  uchar* ptr2_c2=ptr2_c1+img2.elemSize1();

			*ptr_r=*ptr2;
			*ptr_rc1=*ptr2_c1;
			*ptr_rc2=*ptr2_c2;

			ptr_r+=img_result.elemSize();
		}	
	}

	imshow("image_result", img_result);
	cvSaveImage("E:\\final_result.jpg", &IplImage(img_result));
	//	while(1)
	//{
	//	if(waitKey(100)==19) cvSaveImage("E:\\final_result.jpg",  &IplImage(img_result));  
	//	if(waitKey(100)==27) break;                     //按esc退出，ctl+s保存图像
	//}
	system("pause");
	#ifdef _DEBUG
			ofs.flush();
			ofs.close();
			ofs_inline.flush();
			ofs_inline.close();
	#endif // _DEBUG
	return 0;
}

