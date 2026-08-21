#ifndef SHOWPIC_H
#define SHOWPIC_H

#include <QDialog>
#include"thread.h"

namespace Ui {
class ShowPic;
}

class ShowPic : public QDialog
{
    Q_OBJECT

public:
    explicit ShowPic(QDialog *parent = nullptr);
    ~ShowPic();
    void startDetect(const QString& imagePath);  // 传入图片路径，开子线程检测
private slots:
    void onResult(cv::Mat frame, int num);       // 接收子线程发回的标注图和人数

private:
    Ui::ShowPic *ui;
    Thread* m_thread = nullptr;
};

#endif // SHOWPIC_H
