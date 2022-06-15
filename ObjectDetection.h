#ifndef OBJECTDETECTION_H
#define OBJECTDETECTION_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <QPlainTextEdit>

#pragma once

struct Detection
{
    int class_id;
    float confidence;
    cv::Rect box;
};

class ObjectDetection
{
    private:
        const std::vector<cv::Scalar> colors = {cv::Scalar(255, 255, 0), cv::Scalar(0, 0, 255), cv::Scalar(255, 0, 0)};

        std::vector<std::string> load_class_list(std::string class_list_path);

        void load_net(bool is_cuda, std::string model_path);

        cv::Mat format_yolov5(const cv::Mat &source);

    public:
        float input_width, input_height; 
        float score_threshold, nms_threshold, confidence_threshold, classes_score_threshold;
        int num_class;
        cv::dnn::Net net;
        std::vector<std::string> classes_list;

        QPlainTextEdit *ObjDectQPTE;

        ObjectDetection();

        ObjectDetection(std::string model_path, std::string class_list_path, float width, float height, int num_class, bool is_cuda);


        ObjectDetection(std::string model_path, std::string class_list_path, float width, float height, int num_class, bool is_cuda, 
                        float score_threshold, float nms_threshold,float confidence_threshold, float classes_score_threshold);

        ~ObjectDetection();

        void model_init(std::string model_path, std::string class_list_path, float width, float height, int num_class, bool is_cuda, 
                    float score_threshold, float nms_threshold,float confidence_threshold, float classes_score_threshold);

        void detect(cv::Mat &image, std::vector<Detection> &output); 

        cv::Mat draw_bbox(cv::Mat image, std::vector<Detection> &output);
      
};

#endif //OBJECTDETECTION_H
