#include "thread.h"


Thread::Thread(QObject *parent):QThread(parent)
{

}

Thread::~Thread()
{
    stopThread();
    wait();
}

void Thread::stopThread()
{
    m_stop=true;
}

void Thread::setImagePath(const QString &path)
{
     m_imagePath = path;
}

void Thread::run()
{
    std::string modelpath="C:\\Users\\29660\\source\\repos\\Face_Track\\x64\\Release\\models";
    //图片模式
    if(!m_imagePath.isEmpty())
    {
        cv::Mat frame = cv::imread(m_imagePath.toLocal8Bit().toStdString(), cv::IMREAD_COLOR);
        if(frame.empty())
        {
            emit sendFrame(cv::Mat(), -1);//-1表示读取失败
            return;
        }

        MTCNN detector(modelpath);
        detector.setMinFace(40);

        ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_BGR2RGB,
                                                    frame.cols, frame.rows);
        std::vector<Bbox> boxes;
        detector.detect(ncnn_img, boxes);//检测所有人脸

        for(const auto& info : boxes)
        {
            cv::Rect rect((int)info.x1, (int)info.y1,
                          (int)(info.x2 - info.x1 + 1), (int)(info.y2 - info.y1 + 1));
            cv::rectangle(frame, rect, cv::Scalar(0, 255, 0), 2);
            for(int j = 0; j < 5; j++)
            {
                cv::Point p((int)info.ppoint[j], (int)info.ppoint[j + 5]);
                cv::circle(frame, p, 2, cv::Scalar(0, 255, 0), 2);
            }
        }

        emit sendFrame(frame, (int)boxes.size());
        return;
    }
    //摄像头模式
    FaceTracking faceTrack(modelpath);
    if(!m_cap.open(0,cv::CAP_DSHOW))
    {
        return;
    }
    cv::Mat frame;
    m_stop=false;
    int faceIndex=0;
    std::vector<int>IDs;
    std::vector<cv::Scalar>Color;
    srand((unsigned int)time(nullptr));//随机种子

    while(!m_stop)
    {
        if(!m_cap.read(frame))
            continue;
        if(faceIndex==0)
        {
            faceTrack.Init(frame);
            faceIndex=1;
        }
        else
        {
            faceTrack.update(frame);
        }
        //绘制人脸框加关键点
        std::vector<Face>faces=faceTrack.trackingFace;
        cv::Scalar color;
        for(auto& info:faces)
        {
            cv::Rect rect;
            rect.x = info.facebox.x1;
            rect.y = info.facebox.y1;
            rect.width = info.facebox.x2 - info.facebox.x1;
            rect.height = info.facebox.y2 - info.facebox.y1;

            bool isExist = false;
            for (int j = 0; j < IDs.size(); j++)
            {
                if (IDs[j] == info.face_id)
                {
                    color = Color[j];
                    isExist = true;
                    break;
                }
            }
            if (!isExist)
            {
                IDs.push_back(info.face_id);
                int r = rand() % 255 + 1;
                int g = rand() % 255 + 1;
                int b = rand() % 255 + 1;
                color = cv::Scalar(r, g, b);
                Color.push_back(color);
            }
            cv::rectangle(frame, rect, color, 2);
            for (int j = 0; j < 5; j++)
            {
                cv::Point p(info.facebox.ppoint[j], info.facebox.ppoint[j + 5]);
                cv::circle(frame, p, 2, color, 2);
            }
        }

        //发送图像给主线程
        emit sendFrame(frame,faces.size());
        msleep(30);
    }
    m_cap.release();
}
