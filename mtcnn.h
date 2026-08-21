#ifndef __MTCNN_NCNN_H__
#define __MTCNN_NCNN_H__
#include"net.h"
#include<string>
#include<vector>
#include<time.h>
#include<algorithm>
#include<map>
#include<iostream>
struct Bbox
{
	float score;//人脸框置信度
	float x1;
	float y1;
	float x2;
	float y2;
	float regreCoord[4];//偏移修正
	int area;//人脸框面积
	int ppoint[10];
};

class MTCNN
{
public:
	MTCNN(const std::string& model_path);
	MTCNN(const std::vector<std::string>& param_files, const std::vector<std::string>& bin_files);
	~MTCNN();
	
	void setMinFace(int minSize);//设置最小人脸
	void detect(ncnn::Mat& img_, std::vector<Bbox>& finalBbox);
	void detectMaxFace(ncnn::Mat& img_, std::vector<Bbox>&finalBbox);
	float rnet(ncnn::Mat& img);
	Bbox onet(ncnn::Mat& img,int x,int y,int w,int h);
	ncnn::Net Pnet,//粗搜候选
			  Rnet,//筛选假货
			  Onet;//精确定位

private:
	void refine(std::vector<Bbox>& bboxes, const int& height, const int& width, bool square);//边界框回归
	float iou(Bbox& b1, Bbox& b2, std::string modelname = "Union");//nms阈值去除相互重叠的冗余人脸框
	void nms(std::vector<Bbox>& boundingBox_, const float overlap_threshold, std::string modelname = "Union");
	bool cmpScore(Bbox lsh, Bbox rsh);
	bool cmpArea(Bbox lsh, Bbox rsh);
	void generateBbox(ncnn::Mat score, ncnn::Mat location,std::vector<Bbox>& boundingBox_, float scale);//制造box对象
	void nmsTwoBoxs(std::vector<Bbox>& boundingBox_, std::vector<Bbox>& previousBox_, const float overlap_threshold, std::string modelname = "Union");//对比两帧之间的检测结果
	void extractMaxFace(std::vector<Bbox>& boundingBox_);//只保留最大的人脸框
	void smoothBbox(std::vector<Bbox>& winList);//平滑人脸框
	void PNet(float scale);//粗搜候选
	void PNet();
	void RNet();
	void ONet();
private:
	ncnn::Mat img;
	const float nms_threshold[3] = { 0.5f, 0.7f, 0.7f };//nms阈值去除相互重叠的冗余人脸框
	const float mean_vals[3] = { 127.5, 127.5, 127.5 };//图像归一化参数
	const float norm_vals[3] = { 0.0078125, 0.0078125, 0.0078125 };//图像归一化参数
	const int MIN_DET_SIZE = 12;
	std::vector<Bbox>firstPreviousBbox_, secondPreviousBbox_, thirdPreviousBbox_;
	std::vector<Bbox>firstCurrentBbox_, secondCurrentBbox_, thirdCurrentBbox_;
	int img_w, img_h;
	const float threshold[3] = { 0.8f, 0.8f, 0.7f };
	int minsize = 80;
	const float pre_factor = 0.7090f;//图像金字塔缩放比例
	bool smooth = true;//稳定性
    std::vector<Bbox>preBbox_;
};



#endif // !__MTCNN_NCNN_H__


