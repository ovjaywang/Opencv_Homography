#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
using std::vector;

#ifdef _DEBUG
#include <fstream>
#include <iomanip>
using namespace cv;
using namespace std;
#endif // _DEBUG

class HomographyAjustment
{
public:
	HomographyAjustment(void);

	/*处理DLT参数*/

public:
	~HomographyAjustment(void);

	/* 导入同名点数据:像点数据——(x,y)，物方坐标——（X,Y,Z）                                                                     */
	void ImportCognominalPoints(const vector<double> xL,
		const vector<double> yL,
		const vector<double> xR,
		const vector<double> yR);

	/* 由最新的增量，计算最新的参数值                                                                     */
	void UpdateParameters();

	/* 利用共线方程重新计算像点坐标的近似值                                                                     */
	void UpdateApproximateCoordinates();

	/*  间接平差，解算参数增量                                                                   */
	void GetIncrementalValue(int nCount);

	/* 相机检校计算:一定要先导入同名点数据（ImportCognominalPoints），并设置待求参数初值（SetParameterIntialValue）                                                                     */
	bool CameraCalibarion();

	/* 设置待求参数的初值                                                                     */
	//void SetParameterIntialValue(double Xs,double Ys,double Zs,	double phi,double omega,double kappa,double fx,double fy,
	//	double x0,double y0,double K1,double K2,double P1,	double P2);
	void SetParameterIntialValue(Mat H);

	/* 确认迭代是否终止                                                                     */
	bool IsTerminating();

	/* 获取最终的计算值                                                                     */
	void GetFinalResult(vector<double> & parameters);

private:
	/* 待求参数（内外方位元素，焦距，畸变差参数） 的改正值——即每次平差计算后得到的参数的增量（改正）                                                                    */
	vector<double> IncrementalValue;

	vector<double> priorIncrementalValue;
	/* 像点坐标初值向量                                                                     */
	vector<double> xLSet;
	vector<double> yLSet;
	vector<double> xRSet;
	vector<double> yRSet;
	/* 像点坐标近似值向量                                                                     */
	vector<double> x_approximate;
	vector<double> y_approximate;

	set<int> hehe;
	/* 需要解求的参数（相机内外方位元素，焦距，畸变差参数）                                                                     */
	//外方位元素
	//单位：毫米+弧度
	double p0;
	double p1;
	double p2;
	double p3;
	double p4;
	double p5;
	double p6;
	double p7;
	double p8;


	//精度计算
	vector<double> accuracies;

	//初值运算
	double xx0;
	double yy0;
	double qr[12];
#ifdef _DEBUG
	ofstream ofs;

#endif // _DEBUG
	int nCount;
	/* 标记(是否已经导入同名坐标数据，是否已经设置参数初值)                                                                     */
	bool IsCognominalPointsImported;

	bool IsIntialValueSet;
};
