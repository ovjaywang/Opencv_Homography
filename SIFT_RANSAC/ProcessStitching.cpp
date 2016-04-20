#include "ProcessStitching.h"
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
	String strs[2] = { "../TestImage/1-1.JPG", "../TestImage/1-2.JPG" };
//	String strs[2] = { "../TestImage/a1.JPG", "../TestImage/a2.JPG" };
	Stiching st;
	st.ProcessStitching(strs);
	//Findhomography fh;
	//fh.dostiching(strs);
}