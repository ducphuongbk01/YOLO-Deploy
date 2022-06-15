# YOLO-Deploy
Deploy yolov5s with .onnx file

Framework required:
  1. OpenCV 4.5.5
  2. QT5
  3. CUDA + CUDNN if you want to use GPU

Generate 2 file:
  1. yolov5s.onnx: convert from .pt file trained in source code.
  2. classes_list.txt: include all your model classes with correct index.

And change file diection in Camera object constructor.
