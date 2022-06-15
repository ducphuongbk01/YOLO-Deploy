#ifndef CAMERA_H
#define CAMERA_H

#include <opencv2/opencv.hpp>

#include <QThread>
#include <QImage>
#include <QMetaType>
#include <QWidget>

#include <ObjectDetection.h>


class Camera : public QThread
{
    Q_OBJECT
private:
    ObjectDetection *yolo;
    QWidget *parent;
    cv::VideoCapture *cap;
    cv::Mat frame, detect_frame;
    bool isStream = false;
    bool isDetect = false;
    std::vector<Detection> outYolo;
public:
    int width, height;

    Camera(QWidget *parent,
           std::string model_path, std::string class_list_path, int num_class, bool is_cuda, int width, int height,
           float score_threshold, float nms_threshold,float confidence_threshold, float classes_score_threshold);
    void run();
    void stopStream();
    void startDetect();
    void stopDetect();
signals:
    void frameReady(QImage img);
};

#endif // CAMERA_H
