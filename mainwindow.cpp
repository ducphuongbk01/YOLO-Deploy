#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include <opencv2/opencv.hpp>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    bool ret = (bool)cv::cuda::getCudaEnabledDeviceCount();

    this->cam = new Camera(this,
                           "/home/phuongdoan/Code/TestYOLO/yolov5s.onnx", "/home/phuongdoan/Code/TestYOLO/classes_name.txt",
                           3, ret, 640, 480, 0.6, 0.5, 0.5, 0.4);
    connect(this->cam, SIGNAL(frameReady(QImage)), this, SLOT(showImage(QImage)));
}

MainWindow::~MainWindow()
{
    delete ui;
}


void MainWindow::on_btnStart_clicked()
{
    if(this->ui->btnStart->text()=="START STREAM")
    {
        this->ui->btnStart->setText("STOP STREAM");
        this->cam->start();
    }
    else
    {
        this->ui->btnStart->setText("START STREAM");
        this->cam->stopStream();
    }
}


void MainWindow::on_btnStartDetect_clicked()
{
    if(this->ui->btnStartDetect->text()=="START DETECT")
    {
        this->ui->btnStartDetect->setText("STOP DETECT");
        this->cam->startDetect();
    }
    else
    {
        this->ui->btnStartDetect->setText("START DETECT");
        this->cam->stopDetect();
    }
}

void MainWindow::showImage(QImage img)
{
    this->ui->lbStream->setPixmap(QPixmap::fromImage(img).scaled(this->cam->width, this->cam->height));
}
