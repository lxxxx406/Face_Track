#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include<LandmarkTracking.h>
#include<opencv2/opencv.hpp>
#include"thread.h"
#include <QDialog>
#include <QImage>
inline QImage cvMat2QImage(const cv::Mat& mat)
{
    cv::Mat rgb;
    cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
    QImage img((const uchar*)rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);
    return img.copy();
}

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE


class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override;

private slots:
    void on_Capture_clicked();
    void recvFrame(cv::Mat frame,int num);//接收子线程的图像
    void on_CloseCapture_clicked();

    void on_Picture_clicked();

private:
    Ui::MainWindow *ui;
    Thread* m_cameraThread=nullptr;   // 摄像头专用线程
    bool busy=false;
    QString filePath;
};
#endif // MAINWINDOW_H
