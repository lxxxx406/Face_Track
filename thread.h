#ifndef THREAD_H
#define THREAD_H

#include<QThread>
#include<opencv2/opencv.hpp>
#include"LandmarkTracking.h"
#include<QDebug>
class Thread:public QThread
{
    Q_OBJECT
public:

    explicit Thread(QObject *parent = nullptr);
    ~Thread();
    void stopThread();
    void setImagePath(const QString& path);//设置图片路径，非空则走图片检测模式

signals:
    // 发送每一帧图像给主线程UI
    void sendFrame(cv::Mat frame,int num);


protected:
    void run() override;

private:
    cv::VideoCapture m_cap;
    bool m_stop = false;
    QString m_imagePath;//图片模式路径(空=摄像头模式)
};

#endif // THREAD_H
