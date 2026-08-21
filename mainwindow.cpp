#include "mainwindow.h"
#include "ui_mainwindow.h"
#include<QDebug>
#include<QImage.h>
#include <QFileDialog>
#include <QPixmap>
#include <QMessageBox>
#include"showpic.h"
#include <QDir>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    if(m_cameraThread)
    {
        m_cameraThread->stopThread();
        m_cameraThread->wait();
        delete m_cameraThread;
    }
    delete ui;
}

void MainWindow::on_Capture_clicked()
{
    if(m_cameraThread && m_cameraThread->isRunning())
    {
        return;
    }
    qDebug()<<"1";
    m_cameraThread = new Thread(this);
    connect(m_cameraThread, &Thread::sendFrame,
            this, &MainWindow::recvFrame);
    m_cameraThread->start();
}

void MainWindow::recvFrame(cv::Mat frame,int num)
{
    QImage img = cvMat2QImage(frame);
    ui->Image->setPixmap(QPixmap::fromImage(img).scaled(
    ui->Image->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    ui->Count->setText("当前检测人脸数量："+QString::number(num));
}


void MainWindow::on_CloseCapture_clicked()
{
    if(m_cameraThread)
    {
        m_cameraThread->stopThread();
        m_cameraThread->wait();
        delete m_cameraThread;
        m_cameraThread = nullptr;
        ui->Image->clear();
        ui->Count->setText("当前检测人脸数量：");
    }
}



void MainWindow::on_Picture_clicked()
{
    filePath = QFileDialog::getOpenFileName(
        this,
        tr("选择图片"),
        QDir::homePath(), // 默认打开主目录
        tr("图片文件(*.png *.jpg *.jpeg *.bmp *.gif)")
        );

    if(filePath.isEmpty())
    {
        QMessageBox::information(this,"提示","未选择文件");
        return;
    }
    auto* pic = new ShowPic();
    pic->setAttribute(Qt::WA_DeleteOnClose, true);
    pic->show();
    pic->startDetect(filePath);
}

