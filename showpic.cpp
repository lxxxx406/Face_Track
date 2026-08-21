#include "showpic.h"
#include "ui_showpic.h"

ShowPic::ShowPic(QDialog *parent)
    : QDialog(parent)
    , ui(new Ui::ShowPic)
{
    ui->setupUi(this);
}

ShowPic::~ShowPic()
{
    if (m_thread)
        m_thread->wait();
    delete ui;
}
void ShowPic::startDetect(const QString& imagePath)
{
    m_thread = new Thread;   // 无父对象，避免重复删除
    m_thread->setImagePath(imagePath);
    connect(m_thread, &Thread::sendFrame, this, &ShowPic::onResult);
    connect(m_thread, &QThread::finished, m_thread, &QObject::deleteLater);
    connect(m_thread, &QObject::destroyed, this, [this]{ m_thread = nullptr; });
    m_thread->start();
}

void ShowPic::onResult(cv::Mat frame, int num)
{
    if (frame.empty())
    {
        ui->Count->setText("图片读取失败");
        return;
    }
    if(num==0)
    {
        ui->Count->setText("未检测到人脸");
    }
    else
    {
        ui->Count->setText("人脸数量：" + QString::number(num));
    }

    cv::Mat rgb;
    cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
    QImage img((const uchar*)rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);

    // 缩放到窗口里 Image 这个 label 的大小，保持宽高比
    ui->Image->setPixmap(
        QPixmap::fromImage(img.copy()).scaled(
            ui->Image->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}