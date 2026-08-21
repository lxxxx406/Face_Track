#ifndef ZEUSEESFACETRACKING_H
#define ZEUSEESFACETRACKING_H

#include<opencv2/opencv.hpp>
#include"mtcnn.h"
#include"time.h"
#include<cmath>

inline cv::Rect getFaceRect(const std::vector<cv::Point>& pts)//获取人脸关键点的最小外接矩形
{
	if (pts.size() > 1)
	{
		int xmin = pts[0].x;
		int ymin = pts[0].y;
		int xmax = pts[0].x;
		int ymax = pts[0].y;
		for (int i = 1;i < pts.size();i++)
		{
			if (pts[i].x < xmin)
				xmin = pts[i].x;
			if (pts[i].y < ymin)
				ymin = pts[i].y;
			if (pts[i].x > xmax)
				xmax = pts[i].x;
			if (pts[i].y > ymax)
				ymax = pts[i].y;
		}
		return cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin);
	}
}

class Face
{
public:
	Face(int instance_id, cv::Rect rect)
	{
		face_id = instance_id;
		loc = rect;
		isCanShow = true;
	}
	Face()
	{
		isCanShow = false;
	}


	Bbox facebox;//人脸框
	cv::Rect loc;//人脸框
	int face_id = -1;//人脸id
	long frameId = 0;//记录当前帧id
	int ptr_num = 0;//记录连续丢失帧数

	bool isCanShow = false;//是否可以显示
	cv::Mat frame_face_prev;//上一帧人脸图像
	static cv::Rect SquarePadding(cv::Rect facebox, int margin_rows, int margin_cols, bool max_b)//修正为正方形
	{
		int c_x = facebox.x + facebox.width / 2;
		int c_y = facebox.y + facebox.height / 2;
		int large = 0;
		if (max_b)
		{
			large = max(facebox.height, facebox.width) / 2;
		}
		else
		{
			large = min(facebox.height, facebox.width) / 2;
		}
        int x2 = c_x + large;
        int y2 = c_y + large;
        int x1 = c_x - large;
        int y1 = c_y - large;

        // 裁剪到图像范围内
        x1 = max(0, x1);
        y1 = max(0, y1);
        x2 = min(x2, margin_cols - 1);
        y2 = min(y2, margin_rows - 1);

        int w = x2 - x1;
        int h = y2 - y1;

        // 如果不是正方形，取较小边重新居中
        if (w != h)
        {
            int side = min(w, h);
            int new_x = x1 + (w - side) / 2;
            int new_y = y1 + (h - side) / 2;
            return cv::Rect(new_x, new_y, side, side);
        }

        return cv::Rect(x1, y1, w, h);
	}
	static cv::Rect SquarePadding(cv::Rect facebox, int padding)
	{

		int c_x = facebox.x - padding;
		int c_y = facebox.y - padding;
		return cv::Rect(facebox.x - padding, facebox.y - padding, facebox.width + padding * 2, facebox.height + padding * 2);;
	}

	static double getDistance(cv::Point x, cv::Point y)
	{
		return sqrt(pow(x.x - y.x, 2) + pow(x.y - y.y, 2));
	}
	std::vector<cv::Point>faceSequence;//人脸关键点序列
	std::vector<std::vector<float>>attitudeSequence;//人脸姿态序列
};

class FaceTracking
{
public:
	FaceTracking(std::string modelPath)
	{
		this->detector = new MTCNN(modelPath);
		faceMinSize = 70;
		this->detector->setMinFace(faceMinSize);
		detection_Time = -1;
	}

	~FaceTracking()
	{
		delete this->detector;
	}

	void detecting(cv::Mat* image)
	{
		ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(image->data, ncnn::Mat::PIXEL_BGR2RGB, image->cols, image->rows);//塑造ncnn矩阵
		std::vector<Bbox> finalBbox;
		if (isMaxFace)
		{
			detector->detectMaxFace(ncnn_img, finalBbox);
		}
		else
		{
			detector->detect(ncnn_img, finalBbox);
		}
		const int num_box = finalBbox.size();//获取检测到的人脸数量
		std::vector<cv::Rect> bbox(num_box);
		candidateFaces_lock = 1;
		for (int i = 0;i < num_box;i++)
		{
			bbox[i] = cv::Rect(finalBbox[i].x1, finalBbox[i].y1, finalBbox[i].x2 - finalBbox[i].x1+1, finalBbox[i].y2 - finalBbox[i].y1+1);
			bbox[i] = Face::SquarePadding(bbox[i], image->rows, image->cols, true);
			std::shared_ptr<Face> face = std::make_shared<Face>(trackingID, bbox[i]);//用智能指针创建人脸对象
			(*image)(bbox[i]).copyTo(face->frame_face_prev);//存储发哦frame_face_prev
			trackingID += 1;
			candidateFaces.push_back(*face);
		}
		candidateFaces_lock = 0;//释放互斥锁
	}
	void Init(cv::Mat& image)
	{
		ImageHighDP = image.clone();

		trackingID = 0;
		detection_Interval = 200;//每200ms检测一次
		detecting(&image);
		stabilization = false;

	}

	void doingLandmark_onet(cv::Mat& face, Bbox& faceBbox, int zeroadd_x, int zeroadd_y, int stable_satte = 0)//提取5个关键点
	{
		ncnn::Mat in = ncnn::Mat::from_pixels_resize(face.data, ncnn::Mat::PIXEL_BGR, face.cols, face.rows, 48, 48);//读取图像
		faceBbox = detector->onet(in, zeroadd_x, zeroadd_y, face.cols, face.rows);

	}
	
	void tracking_corrfilter(const cv::Mat& frame, const cv::Mat& model, cv::Rect& trackBox, float scale)//在上一帧人脸周围 3 倍窗口搜索人脸新位置
	{
		trackBox.x /= scale;
		trackBox.y /= scale;
		trackBox.height /= scale;
		trackBox.width /= scale;
		int zeroadd_x = 0;
		int zeroadd_y = 0;
		cv::Mat frame_;
		cv::Mat model_;
		cv::resize(frame, frame_, cv::Size(), 1 / scale, 1 / scale);
		cv::resize(model, model_, cv::Size(), 1 / scale, 1 / scale);
		cv::Mat gray;
		cvtColor(frame_, gray, cv::COLOR_RGB2GRAY);
		cv::Mat gray_model;
		cvtColor(model_, gray_model, cv::COLOR_RGB2GRAY);
		cv::Rect searchWindow;
		searchWindow.width = trackBox.width * 3;
		searchWindow.height = trackBox.height * 3;
		searchWindow.x = trackBox.x + trackBox.width * 0.5 - searchWindow.width * 0.5;
		searchWindow.y = trackBox.y + trackBox.height * 0.5 - searchWindow.height * 0.5;
		searchWindow &= cv::Rect(0, 0, frame_.cols, frame_.rows);//矩形交集
        if (searchWindow.width <= 0 || searchWindow.height <= 0 ||
            gray_model.empty() || searchWindow.width < gray_model.cols ||
            searchWindow.height < gray_model.rows)
        {
            return; // 搜索窗口无效，跳过本帧跟踪
        }
		cv::Mat similarity;
		matchTemplate(gray(searchWindow), gray_model, similarity, cv::TM_CCOEFF_NORMED);//匹配人脸
        if (similarity.empty())
            return;
		double mag_r;
		cv::Point point;
		minMaxLoc(similarity, 0, &mag_r, 0, &point);//找到最高相似值
		trackBox.x = point.x + searchWindow.x;
		trackBox.y = point.y + searchWindow.y;
		trackBox.x *= scale;
		trackBox.y *= scale;
		trackBox.height *= scale;
		trackBox.width *= scale;
	}

	bool tracking(cv::Mat& image, Face& face)
	{
		cv::Rect faceROI = face.loc;
		cv::Mat faceROI_Image;
		tracking_corrfilter(image, face.frame_face_prev, faceROI, tpm_scale);//处理中间帧
        // 检查 ROI 是否在图像范围内且宽高有效
        if (faceROI.x < 0 || faceROI.y < 0 ||
            faceROI.width <= 0 || faceROI.height <= 0 ||
            faceROI.x + faceROI.width > image.cols ||
            faceROI.y + faceROI.height > image.rows)
        {
            return false;
        }
		image(faceROI).copyTo(faceROI_Image);

		doingLandmark_onet(faceROI_Image, face.facebox, faceROI.x, faceROI.y, face.frameId>1);

		float sim = face.facebox.score;
		if (sim > 0.1)
		{
			cv::Rect bdbox;
			bdbox.x = face.facebox.x1;
			bdbox.y = face.facebox.y1;
			bdbox.width = face.facebox.x2 - face.facebox.x1;
			bdbox.height = face.facebox.y2 - face.facebox.y1;

			bdbox = Face::SquarePadding(bdbox, static_cast<int>(bdbox.height * -0.05));
			bdbox = Face::SquarePadding(bdbox, image.rows, image.cols, 1);




			face.facebox.x1 = bdbox.x;
			face.facebox.y1 = bdbox.y;
			face.facebox.x2 = bdbox.x + bdbox.width;
			face.facebox.y2 = bdbox.y + bdbox.height;


			face.loc = bdbox;


			image(bdbox).copyTo(face.frame_face_prev);
			face.frameId += 1;
			face.isCanShow = true;

			return true;

		}
		return false;
	}
	void setMask(cv::Mat& image, cv::Rect& rect_mask)
	{
		int height = image.rows;
		int width = image.cols;
		cv::Mat subImage = image(rect_mask);
		subImage.setTo(0);
	}
	void update(cv::Mat& image)
	{
		ImageHighDP = image.clone();
		if (candidateFaces.size() > 0 && !candidateFaces_lock)
		{
			for (int i = 0; i < candidateFaces.size(); i++)
			{
				trackingFace.push_back(candidateFaces[i]);
			}
			candidateFaces.clear();
		}
		for (std::vector<Face>::iterator iter = trackingFace.begin(); iter != trackingFace.end();)
		{
			if (!tracking(image, *iter))
			{
				iter = trackingFace.erase(iter); //追踪失败 则删除此人脸
			}
			else {
				iter++;
			}
		}

		if (detection_Time < 0)
		{
			detection_Time = (double)cv::getTickCount();
		}
		else
		{
			double diff = (double)(cv::getTickCount() - detection_Time) * 1000 / cv::getTickFrequency();//距离上一次执行间隔时间
			if (diff > detection_Interval)
			{
				for (auto& face : trackingFace)
				{
					setMask(ImageHighDP, face.loc);
				}
				detection_Time = (double)cv::getTickCount();
				detecting(&ImageHighDP);
			}

		}
	}
	std::vector<Face>trackingFace;//跟踪的人脸
private:
	cv::Mat ImageHighDP;//高分辨率图像
	int faceMinSize;
	MTCNN* detector;//人脸检测器
	std::vector<Face>candidateFaces;//候选人脸
	bool candidateFaces_lock;// 多线程互斥锁标志，保护candidateFaces读写冲突
	double detection_Time;// 记录上一次执行人脸检测的时间戳
	double detection_Interval;// 检测间隔(秒)，隔多久跑一次MTCNN检测
	int trackingID;
	bool stabilization;// 是否开启人脸框平滑防抖（对应前面smoothBbox）
	int tpm_scale = 2;// 跟踪模块图像缩放系数，跟踪时图像缩小2倍加速
	bool isMaxFace = true;// 是否只跟踪最大人脸
};

#endif // !
