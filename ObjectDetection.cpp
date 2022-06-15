#include "ObjectDetection.h"

std::vector<std::string> ObjectDetection::load_class_list(std::string class_list_path)
{
    std::vector<std::string> class_list;
    std::ifstream ifs(class_list_path);
    std::string line;
    while (getline(ifs, line))
    {
        class_list.push_back(line);
    }
    return class_list;
}

void ObjectDetection::load_net(bool is_cuda, std::string model_path)
{
    auto result = cv::dnn::readNet(model_path);
    if (is_cuda)
    {
        std::cout << "Attempty to use CUDA\n";
//        ControlLib::AppOutput("Serial Connected",ObjDectQPTE);
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }
    else
    {
        std::cout << "Running on CPU\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    this->net = result;
}

cv::Mat ObjectDetection::format_yolov5(const cv::Mat &source)
{
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

ObjectDetection::ObjectDetection()
{
    ObjDectQPTE = new QPlainTextEdit;
}

ObjectDetection::ObjectDetection(std::string model_path, std::string class_list_path, float width, float height, int num_class, bool is_cuda)
{
    this->num_class = num_class;
    this->input_width = width;
    this->input_height = height;
    this->load_net(is_cuda, model_path);
    this->classes_list = this->load_class_list(class_list_path);
    this->score_threshold = 0.2; 
    this->nms_threshold = 0.4; 
    this->confidence_threshold = 0.4; 
    this->classes_score_threshold = 0.25;

    ObjDectQPTE = new QPlainTextEdit;
}

ObjectDetection::ObjectDetection(std::string model_path, std::string class_list_path, float width, float height, int num_class, bool is_cuda, 
                                float score_threshold, float nms_threshold,float confidence_threshold, float classes_score_threshold)
{
    this->num_class = num_class;
    this->input_width = width;
    this->input_height = height;
    this->load_net(is_cuda, model_path);
    this->classes_list = this->load_class_list(class_list_path);
    this->score_threshold = score_threshold; 
    this->nms_threshold = nms_threshold; 
    this->confidence_threshold = confidence_threshold; 
    this->classes_score_threshold = classes_score_threshold;

    ObjDectQPTE = new QPlainTextEdit;
}

ObjectDetection::~ObjectDetection()
{

}

void ObjectDetection::model_init(std::string model_path, std::string class_list_path, float width, float height, int num_class, bool is_cuda, 
                            float score_threshold, float nms_threshold,float confidence_threshold, float classes_score_threshold)
{
    this->num_class = num_class;
    this->input_width = width;
    this->input_height = height;
    this->load_net(is_cuda, model_path);
    this->classes_list = this->load_class_list(class_list_path);
    this->score_threshold = score_threshold; 
    this->nms_threshold = nms_threshold; 
    this->confidence_threshold = confidence_threshold; 
    this->classes_score_threshold = classes_score_threshold;           
}

void ObjectDetection::detect(cv::Mat &image, std::vector<Detection> &output)
{
    cv::Mat blob;

    auto input_image = this->format_yolov5(image);

    cv::dnn::blobFromImage(input_image, blob, 1./255., cv::Size(this->input_width, this->input_height), cv::Scalar(), true, false);
    this->net.setInput(blob);
    std::vector<cv::Mat> outputs;
    this->net.forward(outputs, this->net.getUnconnectedOutLayersNames());

    float x_factor = input_image.cols / this->input_width;
    float y_factor = input_image.rows / this->input_height;

    float *data = (float *)outputs[0].data; 

    const int rows = 25200;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i) {

        float confidence = data[4];
        if (confidence >= this->confidence_threshold) {

            float * classes_scores = data + 5;
            cv::Mat scores(1, this->classes_list.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            if (max_class_score > this->score_threshold) {

                confidences.push_back(confidence);

                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];
                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                boxes.push_back(cv::Rect(left, top, width, height));
            }

        }

        data += (this->num_class + 5);
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, this->score_threshold, this->nms_threshold, nms_result);
    output.clear();
    for (int i = 0; i < nms_result.size(); i++) {
        int idx = nms_result[i];
        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        output.push_back(result);
    }
}

cv::Mat ObjectDetection::draw_bbox(cv::Mat image, std::vector<Detection> &output)
{
    int detections = output.size();
    cv::Mat detect_frame = image.clone();

    for (int i = 0; i < detections; ++i)
    {
        auto detection = output[i];
        auto box = detection.box;
        auto classId = detection.class_id;
        auto conf = detection.confidence;
        const auto color = colors[classId % colors.size()];
        cv::rectangle(detect_frame, box, color, 3);

        cv::rectangle(detect_frame, cv::Point(box.x, box.y - 25), cv::Point(box.x + box.width, box.y), color, cv::FILLED);
        cv::putText(detect_frame, this->classes_list[classId].c_str(), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
        cv::putText(detect_frame, std::to_string(int(round(conf*100))/100.0).c_str(), cv::Point(int(box.x + box.width/2), box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
    };
    return detect_frame;
}
