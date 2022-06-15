#include "camera.h"
#include <QMessageBox>
#include <QPixmap>

Camera::Camera(QWidget *parent,
               std::string model_path, std::string class_list_path, int num_class, bool is_cuda, int width, int height,
               float score_threshold, float nms_threshold,float confidence_threshold, float classes_score_threshold)
{
    this->parent = parent;
    this->yolo = new ObjectDetection(model_path, class_list_path, width, width, num_class, is_cuda,
                                     score_threshold, nms_threshold, confidence_threshold, classes_score_threshold);

    this->cap = new cv::VideoCapture(0);

//    this->width = this->cap->get(cv::CAP_PROP_FRAME_WIDTH);
//    this->height = this->cap->get(cv::CAP_PROP_FRAME_HEIGHT);

    this->width = 640;
    this->height = 480;


}

void Camera::run()
{
    if(!this->cap->isOpened())
    {
        QMessageBox::warning(parent, "Warning", "Cannot connect to camera.");
    }
    else
    {
        this->isStream = true;

        while(this->isStream)
        {
            this->outYolo.clear();

            *cap >> frame;

            cv::Mat rgb_Frame;
            cv::cvtColor(frame, rgb_Frame, cv::COLOR_BGR2RGB);

            if(this->isDetect)
            {
                this->yolo->detect(rgb_Frame, this->outYolo);

                std::cout << "Number of object: " << this->outYolo.size() << std::endl;

                this->detect_frame = this->yolo->draw_bbox(rgb_Frame, this->outYolo);
            }
            else
            {
                this->detect_frame = rgb_Frame.clone();
            }

            QImage img = QImage((const unsigned char*)detect_frame.data, detect_frame.cols, detect_frame.rows, QImage::Format_RGB888);
            Q_EMIT frameReady(img);
        }
    }

}

void Camera::stopStream()
{
    this->isStream = false;
}

void Camera::startDetect()
{
    this->isDetect = true;
}

void Camera::stopDetect()
{
    this->isDetect = false;
}
