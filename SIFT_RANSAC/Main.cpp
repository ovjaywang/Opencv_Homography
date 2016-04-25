#include "Main.h"

using namespace cv;
ProcessStitching::ProcessStitching()
{
}


ProcessStitching::~ProcessStitching()
{

}

int main(){
	//Findhomography m_homo;
	//m_homo.dostiching();
	String strs[2] = { "../TestImage/2-1.JPG", "../TestImage/2-2.JPG" };
	//	String strs[2] = { "../TestImage/1-1.JPG", "../TestImage/1-2.JPG" };
	//String strs[2] = { "../TestImage/a1.JPG", "../TestImage/a2.JPG" };
	double width = 0.0;
	double up_height = 0.0;
	double down_height = 0.0;

	Stiching st;
	st.ProcessStitching(strs, width,up_height,down_height);

	//Findhomography fh;
	//fh.dostiching(strs);CStdioFile fileReader;

	system("pause");
	return 0;
}