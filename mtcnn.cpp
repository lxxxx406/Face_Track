#include "mtcnn.h"
#include<cmath>


MTCNN::MTCNN(const std::string& model_path)
{
	std::vector<std::string> param_files = {
		model_path + "/det1.param",
		model_path + "/det2.param",
		model_path + "/det3.param"
	};
	std::vector<std::string> bin_files = { 
		model_path + "/det1.bin", 
		model_path + "/det2.bin", 
		model_path + "/det3.bin" 
	};
	Pnet.load_param(param_files[0].data());
	Pnet.load_model(bin_files[0].data());
	Rnet.load_param(param_files[1].data());
	Rnet.load_model(bin_files[1].data());
	Onet.load_param(param_files[2].data());
	Onet.load_model(bin_files[2].data());

}

MTCNN::MTCNN(const std::vector<std::string>& param_files, const std::vector<std::string>& bin_files)
{
	Pnet.load_param(param_files[0].data());
	Pnet.load_model(bin_files[0].data());
	Rnet.load_param(param_files[1].data());
	Rnet.load_model(bin_files[1].data());
	Onet.load_param(param_files[2].data());
	Onet.load_model(bin_files[2].data());
}

MTCNN::~MTCNN()
{
	Pnet.clear();
	Rnet.clear();
	Onet.clear();
}

void MTCNN::refine(std::vector<Bbox>& bboxes, const int& height, const int& width, bool square)
{
	if (bboxes.empty())
	{
		return;
	}
	float bbw = 0, bbh = 0,maxSide=0;

	int w = 0, h = 0;
	float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
	for (auto it = bboxes.begin(); it != bboxes.end(); it++)
	{
		bbw = (it->x2 - it->x1 + 1);
		bbh = (it->y2 - it->y1 + 1);
		//偏移修正
		x1 = it->x1 + it->regreCoord[0] * bbw;
		y1 = it->y1 + it->regreCoord[1] * bbh;
		x2 = it->x2 + it->regreCoord[2] * bbw;
		y2 = it->y2 + it->regreCoord[3] * bbh;

		if (square)//将人脸框调整为正方形
		{
			w = x2 - x1 + 1;
			h = y2 - y1 + 1;
			maxSide = (h > w) ? h : w;
			x1 = x1 + w * 0.5 - maxSide * 0.5;
			y1 = y1 + h * 0.5 - maxSide * 0.5;
			(*it).x2 = round(x1 + maxSide - 1);
			(*it).y2 = round(y1 + maxSide - 1);
			(*it).x1 = round(x1);
			(*it).y1 = round(y1);
		}
		//边界检查
		if(it->x1<0)
		{
			it->x1 = 0;
		}
		if(it->y1<0)
		{
			it->y1 = 0;
		}
		if(it->x2>=width)
		{
			it->x2 = width - 1;
		}
		if(it->y2>=height)
		{
			it->y2 = height - 1;
		}
		it->area = (it->x2 - it->x1) * (it->y2 - it->y1);
	}
}

void MTCNN::setMinFace(int minSize)
{
	minsize = minSize;
}

void MTCNN::detect(ncnn::Mat& img_, std::vector<Bbox>& finalBbox)
{
	img = img_;
	img_w = img.w;
	img_h = img.h;
	img.substract_mean_normalize(mean_vals, norm_vals);
	//第一层粗筛
	PNet();
	if (firstCurrentBbox_.empty())
	{
		return;
	}
	nms(firstCurrentBbox_, nms_threshold[0]);//nms阈值去除相互重叠的冗余人脸框
	refine(firstCurrentBbox_, img_h, img_w, true);//边界框回归

	//第二层筛选
	RNet();
	if (secondCurrentBbox_.empty())
	{
		return;
	}
	nms(secondCurrentBbox_, nms_threshold[1]);//nms阈值去除相互重叠的冗余人脸框
	refine(secondCurrentBbox_, img_h, img_w, true);//边界框回归

	//第三层精确定位
	ONet();
	if (thirdCurrentBbox_.empty())
	{
		return;
	}
	refine(thirdCurrentBbox_, img_h, img_w, true);//边界框回归
	nms(thirdCurrentBbox_, nms_threshold[2], "Min");//nms阈值去除相互重叠的冗余人脸框
	finalBbox = thirdCurrentBbox_;
	if (smooth)
	{
		smoothBbox(finalBbox);
	}

}

void MTCNN::detectMaxFace(ncnn::Mat& img_, std::vector<Bbox>& finalBbox)
{
	firstPreviousBbox_.clear();
	secondPreviousBbox_.clear();
	thirdPreviousBbox_.clear();
	firstCurrentBbox_.clear();
	secondCurrentBbox_.clear();
	thirdCurrentBbox_.clear();

	//检测最大人脸
	img = img_;
	img_w = img.w;
	img_h = img.h;
	img.substract_mean_normalize(mean_vals, norm_vals);//归一化

	float minl = img_w < img_h ? (float)img_w : (float)img_h;	
	float m = (float)MIN_DET_SIZE / minsize;
	minl *= m;
	float factor = pre_factor;
	std::vector<float>scales_;
	while (minl > MIN_DET_SIZE)
	{
		scales_.push_back(m);
		minl *= factor;
		m *= factor;
	}
	sort(scales_.begin(), scales_.end());
	for (size_t i = 0;i < scales_.size();i++)
	{
		//第一次初筛
		PNet(scales_[i]);
		nms(firstCurrentBbox_, nms_threshold[0]);//nms阈值去除相互重叠的冗余人脸框
		nmsTwoBoxs(firstCurrentBbox_, firstPreviousBbox_, nms_threshold[0]);//对比两帧之间的检测结果
		if (firstCurrentBbox_.empty())
		{
			firstCurrentBbox_.clear();
			continue;
		}
		firstPreviousBbox_.insert(firstPreviousBbox_.end(), firstCurrentBbox_.begin(), firstCurrentBbox_.end());
		refine(firstPreviousBbox_, img_h, img_w, true);//边界框回归

		//第二次筛选
		RNet();
		nms(secondCurrentBbox_, nms_threshold[1]);//nms阈值去除相互重叠的冗余人脸框
		nmsTwoBoxs(secondCurrentBbox_, secondPreviousBbox_, nms_threshold[1]);//对比两帧之间的检测结果
		if (secondCurrentBbox_.empty())
		{
			firstCurrentBbox_.clear();
			secondCurrentBbox_.clear();
			continue;
		}
		secondPreviousBbox_.insert(secondPreviousBbox_.end(), secondCurrentBbox_.begin(), secondCurrentBbox_.end());
		refine(secondPreviousBbox_, img_h, img_w, true);//边界框回归

		//第三次精确定位
		ONet();
		if (thirdCurrentBbox_.empty())
		{
			firstCurrentBbox_.clear();
			secondCurrentBbox_.clear();
			thirdCurrentBbox_.clear();
			continue;
		}
		refine(thirdCurrentBbox_, img_h, img_w, true);//边界框回归
		nms(thirdCurrentBbox_, nms_threshold[2], "Min");//nms阈值去除相互重叠的冗余人脸框

		if (thirdCurrentBbox_.size() > 0)
		{
			extractMaxFace(thirdCurrentBbox_);
			finalBbox = thirdCurrentBbox_;
			if (smooth)
			{
				smoothBbox(finalBbox);
			}
			break;
		}
	}
}

float MTCNN::rnet(ncnn::Mat& img)
{
	ncnn::Extractor ex = Rnet.create_extractor();
	const float mean_vals[3] = { 127.5f, 127.5f, 127.5f };//图像归一化参数
	const float norm_vals[3] = { 1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5 };//图像归一化参数
	img.substract_mean_normalize(mean_vals, norm_vals);//归一化
	ex.set_light_mode(true);
	ex.input("data", img);
	ncnn::Mat score_;
	ex.extract("prob1", score_);
	return (float)score_[1];
}

Bbox MTCNN::onet(ncnn::Mat& img, int x, int y, int w, int h)
{
	Bbox facebox;
	const float mean_vals[3] = { 127.5f, 127.5f, 127.5f };//图像归一化参数
	const float norm_vals[3] = { 1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5 };//图像归一化参数
	img.substract_mean_normalize(mean_vals, norm_vals);//归一化
	ncnn::Extractor ex = Onet.create_extractor();

	ex.set_light_mode(true);
	ex.input("data", img);
	ncnn::Mat score_, location_, keyPoint_;
	ex.extract("prob1", score_);
	ex.extract("conv6-2", location_);
	ex.extract("conv6-3", keyPoint_);
	facebox.score = (float)score_[1];
	facebox.x1 = static_cast<int>( w * location_[0])+x;
	facebox.y1 = static_cast<int>( h * location_[1])+y;
	facebox.x2 = static_cast<int>( w * location_[2])+w+x;
	facebox.y2 = static_cast<int>( h * location_[3])+h+y;
	for (int num = 0;num < 5;num++)//关键点坐标
	{
		(facebox.ppoint)[num] = x + w * keyPoint_[num];
		(facebox.ppoint)[num + 5] = y + h * keyPoint_[num + 5];
	}
	return facebox;
}

float MTCNN::iou(Bbox& b1, Bbox& b2, std::string modelname)
{
	float IOU = 0;
	float maxX = 0,maxY=0,minX=0,minY=0;
	maxX = max(b1.x1, b2.x1);
	maxY = max(b1.y1, b2.y1);
	minX = min(b1.x2, b2.x2);
	minY = min(b1.y2, b2.y2);
	//判断相交
	maxX = ((minX - maxX + 1) > 0) ? maxX : 0;
	maxY = ((minY - maxY + 1) > 0) ? maxY : 0;
	IOU = maxX * maxY;
	//计算IOU实际交集面积
	if (!modelname.compare("Union"))
	{
		IOU = IOU / (b1.area + b2.area - IOU);
	}
	else if (!modelname.compare("Min"))
	{
		IOU = IOU / ((b1.area < b2.area) ? b1.area : b2.area);
	}
	return IOU;
}

void MTCNN::nms(std::vector<Bbox>& boundingBox_, const float overlap_threshold, std::string modelname)
{
	if (boundingBox_.empty())
	{
		return;
	}
	sort(boundingBox_.begin(), boundingBox_.end(), [](const Bbox& a, const Bbox& b)
		{
			return a.score < b.score;
		});
	float IOU = 0;
	float maxX = 0, maxY = 0, minX = 0, minY = 0;
	std::vector<int>vPick;//人脸框下标
	int nPick = 0;//记录有效数目
	std::multimap<float, int> vScores;
	const int num_boxes = boundingBox_.size();
	vPick.resize(num_boxes);
	for (int i = 0; i < num_boxes; ++i)
	{
		vScores.insert(std::pair<float, int>(boundingBox_[i].score, i));
	}
	while (vScores.size() > 0)
	{
		int last = vScores.rbegin()->second;
		vPick[nPick] = last;
		nPick++;
		//逐个判断重合
		for (std::multimap<float, int>::iterator it = vScores.begin(); it != vScores.end();) {
			int it_idx = it->second;
			maxX = max(boundingBox_.at(it_idx).x1, boundingBox_.at(last).x1);
			maxY = max(boundingBox_.at(it_idx).y1, boundingBox_.at(last).y1);
			minX = min(boundingBox_.at(it_idx).x2, boundingBox_.at(last).x2);
			minY = min(boundingBox_.at(it_idx).y2, boundingBox_.at(last).y2);
			//maxX1 and maxY1 reuse 
			maxX = ((minX - maxX + 1) > 0) ? (minX - maxX + 1) : 0;
			maxY = ((minY - maxY + 1) > 0) ? (minY - maxY + 1) : 0;
			//IOU reuse for the area of two bbox
			IOU = maxX * maxY;
			if (!modelname.compare("Union"))
				IOU = IOU / (boundingBox_.at(it_idx).area + boundingBox_.at(last).area - IOU);
			else if (!modelname.compare("Min")) {
				IOU = IOU / ((boundingBox_.at(it_idx).area < boundingBox_.at(last).area) ? boundingBox_.at(it_idx).area : boundingBox_.at(last).area);
			}
			if (IOU > overlap_threshold) {
				it = vScores.erase(it);
			}
			else {
				it++;
			}
		}
	}
	//保留剩下人脸
	vPick.resize(nPick);
	std::vector<Bbox>tmp_;
	tmp_.resize(nPick);
	for (int i = 0; i < nPick; i++)
	{
		tmp_[i] = boundingBox_[vPick[i]];
	}
	boundingBox_ = tmp_;
}

bool MTCNN::cmpScore(Bbox lsh,Bbox rsh)
{
	if(lsh.score<rsh.score)
	{
		return true;
	}
	else
	{
		return false;
	}
}

bool MTCNN::cmpArea(Bbox lsh, Bbox rsh)
{
	if (lsh.area < rsh.area)
	{
		return false;
	}
	else
	{
		return true;
	}
}

void MTCNN::generateBbox(ncnn::Mat score, ncnn::Mat location, std::vector<Bbox>& boundingBox_, float scale)
{
	const int stride = 2;//步长
	const int cellsize = 12;//Pnet输入图像大小
	float* p = score.channel(1);//置信度 score.data+score.cstep
	Bbox bbox;
	float inv_scale = 1.0f / scale;
	for (int row = 0; row < score.h; row++)
	{
		for (int col = 0; col < score.w; col++)
		{
			if (*p > threshold[0])//存在人脸框
			{
				bbox.score = *p;
				bbox.x1 = round((stride * col + 1) * inv_scale);
				bbox.y1 = round((stride * row + 1) * inv_scale);
				bbox.x2 = round((stride * col + cellsize) * inv_scale);
				bbox.y2 = round((stride * row + cellsize) * inv_scale);
				bbox.area = (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1);
				for (int channel = 0; channel < 4; channel++)
				{
					bbox.regreCoord[channel] = location.channel(channel)[row * score.w + col];//location中的偏移回归值储存在box中
				}
				boundingBox_.push_back(bbox);
			}
			p++;
		}
	}
}

void MTCNN::nmsTwoBoxs(std::vector<Bbox>& boundingBox_, std::vector<Bbox>& previousBox_, const float overlap_threshold, std::string modelname)
{
	if (boundingBox_.empty())
	{
		return;
	}
	
	sort(boundingBox_.begin(), boundingBox_.end(), 
		[](const Bbox& a, const Bbox& b)
		{
			return a.score < b.score;	
		});
	for (std::vector<Bbox>::iterator it = previousBox_.begin(); it != previousBox_.end();it++)
	{
		for (std::vector<Bbox>::iterator it2 = boundingBox_.begin(); it2 != boundingBox_.end();)
		{
			float IOU = iou(*it, *it2, modelname);
			if (IOU > overlap_threshold && it2->score > it->score)
			{
				it2 = boundingBox_.erase(it2);
			}
			else
			{
				it2++;
			}
		}
		
	}
	std::cout << "boundingBox_ size: " << boundingBox_.size() << std::endl;
}

void MTCNN::extractMaxFace(std::vector<Bbox>& boundingBox_)
{
	if (boundingBox_.empty())
	{
		return;
	}
	sort(boundingBox_.begin(), boundingBox_.end(), [](const Bbox& a, const Bbox& b)
		{
			return a.area > b.area;
		});
	for (auto it = boundingBox_.begin()+1; it != boundingBox_.end();)
	{
		it = boundingBox_.erase(it);
	}
}

void MTCNN::smoothBbox(std::vector<Bbox>& winList)
{
	for (int i = 0; i < winList.size(); i++)
	{
		for (int j = 0; j < preBbox_.size(); j++)
		{
			float IOU = iou(winList[i], preBbox_[j], "Union");
			if (IOU > 0.85)
			{
				winList[i].x1 = (winList[i].x1 + preBbox_[j].x1) / 2;
				winList[i].y1 = (winList[i].y1 + preBbox_[j].y1) / 2;
				winList[i].x2 = (winList[i].x2 + preBbox_[j].x2) / 2;
				winList[i].y2 = (winList[i].y2 + preBbox_[j].y2) / 2;
				winList[i].score = (winList[i].score + preBbox_[j].score) / 2;
				for (int k = 0;k < 10;k++)
				{
					winList[i].ppoint[k] = (winList[i].ppoint[k] + preBbox_[j].ppoint[k]) / 2;
				}
			}
		}
	}
	preBbox_ = winList;
}

void MTCNN::PNet(float scale)
{
	int hs = (int)ceil(img_h * scale);
	int ws = (int)ceil(img_w * scale);
	ncnn::Mat in;
	resize_bilinear(img, in, ws, hs);//缩放图像
	ncnn::Extractor ex = Pnet.create_extractor(); //创建提取器
	ex.set_light_mode(true);//设置轻量模式
	ex.input("data", in);//输入图像
	ncnn::Mat score_, location_;
	ex.extract("prob1", score_);//置信度
	ex.extract("conv4-2", location_);//偏移回归
	std::vector<Bbox> boundingBox_;

	generateBbox(score_, location_, boundingBox_, scale);//生成候选人脸框
	nms(boundingBox_, nms_threshold[0]);//非极大值抑制

	firstCurrentBbox_.insert(firstCurrentBbox_.end(), boundingBox_.begin(), boundingBox_.end());//将当前帧的候选人脸框加入到firstCurrentBbox_中
	boundingBox_.clear();
}

void MTCNN::PNet()
{
	firstCurrentBbox_.clear();
	float minl = img_w < img_h ? img_w : img_h;
	float m = (float)MIN_DET_SIZE / minsize;
	minl *= m;
	float factor = pre_factor;
	std::vector < float> scales_;
	while (minl > MIN_DET_SIZE)//图像金字塔缩放
	{
		scales_.push_back(m);
		minl *= factor;
		m *= factor;
	}
	for (size_t i = 0;i < scales_.size(); i++)
	{
		PNet(scales_[i]);
	}
}

void MTCNN::RNet()
{
	secondCurrentBbox_.clear();
	int count = 0;
	for (auto it = firstCurrentBbox_.begin(); it != firstCurrentBbox_.end(); it++)
	{
		ncnn::Mat tempIm;//裁剪人脸框
		copy_cut_border(img, tempIm, it->y1, img_h - it->y2, it->x1, img_w - it->x2);
		ncnn::Mat in;
		resize_bilinear(tempIm, in, 24, 24);//缩放人脸框
		ncnn::Extractor ex = Rnet.create_extractor();
		ex.set_light_mode(true);
		ex.input("data", in);
		ncnn::Mat score_, location_;
		ex.extract("prob1", score_);
		ex.extract("conv5-2", location_);
		if ((float)score_[1] > threshold[1])
		{
			for (int channel = 0;channel < 4; channel++)
			{
				it->regreCoord[channel] = location_[channel];//location_中的偏移回归值储存在box中
			}
			it->area = (it->x2 - it->x1) * (it->y2 - it->y1);
			it->score = score_.channel(1)[0];//取出通道1的人脸得分
			secondCurrentBbox_.push_back(*it);//将符合条件的人脸框加入到secondCurrentBbox_中
		}
	}
}

void MTCNN::ONet()
{
	thirdCurrentBbox_.clear();
	for (auto it = secondCurrentBbox_.begin(); it != secondCurrentBbox_.end(); it++)
	{
		ncnn::Mat tempIm;
		copy_cut_border(img, tempIm, it->y1, img_h - it->y2, it->x1, img_w - it->x2);
		ncnn::Mat in;
		resize_bilinear(tempIm, in, 48, 48);
		ncnn::Extractor ex = Onet.create_extractor();
		ex.set_light_mode(true);
		ex.input("data", in);
		ncnn::Mat score_, location_, keyPoint_;
		ex.extract("prob1", score_);
		ex.extract("conv6-2", location_);
		ex.extract("conv6-3", keyPoint_);//关键点坐标
		if ((float)score_[1] > threshold[2])
		{
			for (int channel = 0;channel < 4; channel++)
			{
				it->regreCoord[channel] = location_[channel];
			}
			for (int num = 0;num < 5; num++)//关键点坐标
			{
				it->ppoint[num] = it->x1 + (it->x2 - it->x1) * keyPoint_[num];
				it->ppoint[num + 5] = it->y1 + (it->y2 - it->y1) * keyPoint_[num + 5];
			}
			it->area = (it->x2 - it->x1) * (it->y2 - it->y1);
			it->score = score_.channel(1)[0];
			thirdCurrentBbox_.push_back(*it);
		}
	}
}
